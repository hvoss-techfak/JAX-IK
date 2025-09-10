import jax.numpy as jnp
import numpy as np
import pytest

from jax_ik.objectives import (
    BoneDirectionObjective,
    BoneRelativeLookObj,
    BoneZeroRotationObj,
    CombinedDerivativeObj,
    DerivativeObj,
    DistanceObjTraj,
    EqualDistanceObj,
    InitPoseObj,
    SDFCollisionPenaltyObj,
    SDFSelfCollisionPenaltyObj,
    SphereCollisionPenaltyObjTraj,
)
from jax_ik.smplx_statics import left_arm_bounds_dict


@pytest.fixture()
def fake_sdf():
    # Small 5^3 grid positive outside, negative pocket in center to simulate penetration
    grid = jnp.ones((5, 5, 5), dtype=jnp.float32) * 0.02
    grid = grid.at[2, 2, 2].set(-0.01)
    origin = jnp.array([-0.1, -0.1, -0.1], dtype=jnp.float32)
    spacing = jnp.float32(0.05)
    return {"grid": grid, "origin": origin, "spacing": spacing}


@pytest.fixture()
def solver_with_fake_sdf(solver, fake_sdf):
    # Attach simple mesh + sdf for self-collision objectives
    fk_solver = solver.fk_solver
    if fk_solver.sdf is None:
        fk_solver.sdf = fake_sdf
    if fk_solver.mesh_data is None:
        # minimal mesh_data with only vertices + faces so inverse_skin_points early-outs
        verts = np.array(
            [
                [0, 0, 0, 1],
                [0.01, 0, 0, 1],
                [0, 0.01, 0, 1],
                [0, 0, 0.01, 1],
                [0.01, 0.01, 0, 1],
                [0.01, 0, 0.01, 1],
                [0, 0.01, 0.01, 1],
                [0.01, 0.01, 0.01, 1],
            ],
            dtype=np.float32,
        )
        fk_solver.mesh_data = {
            "vertices": jnp.asarray(verts),
            "faces": np.array([[0, 1, 2]]),
        }
    return solver


def _run_solver_once(solver, mandatory=None, optional=None, traj=False):
    from jax_ik.objectives import DistanceObjTraj

    init = np.zeros(len(solver.controlled_bones) * 3, dtype=np.float32)
    if not mandatory:
        # Provide a small positive weight distance objective to encourage numerical stability
        mandatory = [
            DistanceObjTraj(
                bone_name="left_wrist",
                target_points=[[0.01, 0.01, 0.01]],
                use_head=True,
                weight=0.05,
            )
        ]
    res, loss, steps = solver.solve(
        initial_rotations=init,
        mandatory_objective_functions=mandatory or [],
        optional_objective_functions=optional or [],
        ik_points=1 if not traj else 2,
        learning_rate=0.15,
        patience=20,
        verbose=False,
    )
    # After nan guards, loss should always be finite
    assert np.isfinite(loss)
    return res, loss, steps


def test_distance_obj_and_update(solver):
    t1 = [0.0, 0.0, 0.3]
    obj = DistanceObjTraj(
        bone_name="left_index3", target_points=[t1], use_head=True, weight=1.0
    )
    res1, loss1, _ = _run_solver_once(solver, mandatory=[obj])
    params = obj.get_params()
    params["weight"] = 1.0
    params["target_points"] = [[0.1, 0.2, 0.5]]
    obj.update_params(params)
    res2, loss2, _ = _run_solver_once(solver, mandatory=[obj])
    # Compare normalized losses (divide by weight). Allow some tolerance for target change
    assert loss1 < 1e-3, "DistanceObjTraj did not converge sufficiently to first target"
    assert loss2 < 1e-3, (
        "DistanceObjTraj did not converge sufficiently to second target"
    )


def test_bone_direction_obj(solver):
    obj = BoneDirectionObjective(
        bone_name="left_elbow", directions=[[0, 1, 0]], weight=0.5
    )
    _run_solver_once(solver, mandatory=[obj])
    p = obj.get_params()
    p["weight"] = 1.0
    obj.update_params(p)
    _run_solver_once(solver, mandatory=[obj])


def test_bone_zero_rotation(solver):
    obj = BoneZeroRotationObj(weight=0.1)
    _run_solver_once(solver, optional=[obj])
    p = obj.get_params()
    p["weight"] = 0.2
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj])


def test_init_pose_obj(solver):
    init_pose = np.zeros(len(solver.controlled_bones) * 3, dtype=np.float32)
    mask = np.zeros_like(init_pose)
    mask[0:3] = 1.0
    obj = InitPoseObj(init_rot=init_pose, last_position=True, weight=0.3, mask=mask)
    _run_solver_once(solver, optional=[obj])
    p = obj.get_params()
    p["weight"] = 0.6
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj])


def test_equal_distance_obj(solver):
    obj = EqualDistanceObj(weight=0.2)
    _run_solver_once(solver, optional=[obj], traj=True)
    p = obj.get_params()
    p["weight"] = 0.4
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj], traj=True)


def test_derivative_obj_orders(solver):
    for order in (1, 2, 3):
        obj = DerivativeObj(order=order, weight=0.05)
        _run_solver_once(solver, optional=[obj], traj=True)
        p = obj.get_params()
        p_weight = p["weight"]
        p["weight"] = p_weight * 2
        obj.update_params(p)
        _run_solver_once(solver, optional=[obj], traj=True)


def test_combined_derivative_obj(solver):
    obj = CombinedDerivativeObj(max_order=3, weights=[0.01, 0.02, 0.03])
    _run_solver_once(solver, optional=[obj], traj=True)
    p = obj.get_params()
    p["weights"] = [0.02, 0.02, 0.02]
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj], traj=True)


def test_bone_relative_look_obj(solver):
    mods = [(0, 0.01), (1, -0.01)]
    obj = BoneRelativeLookObj(
        bone_name="left_elbow", use_head=True, modifications=mods, weight=0.5
    )
    _run_solver_once(solver, optional=[obj])
    p = obj.get_params()
    p["modifications"] = [(0, 0.02)]
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj])


def test_sphere_collision_penalty(solver):
    collider = {"center": [0.0, 0.0, 0.0], "radius": 0.001}
    obj = SphereCollisionPenaltyObjTraj(
        sphere_collider=collider, weight=0.2, min_clearance=0.0
    )
    _run_solver_once(solver, optional=[obj], traj=True)
    p = obj.get_params()
    p["weight"] = 0.4
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj], traj=True)


def test_sdf_collision_penalty(fake_sdf, solver):
    # Provide SDF explicitly (FK path only)
    obj = SDFCollisionPenaltyObj(
        bone_name="left_wrist", sdf=fake_sdf, num_samples=3, weight=0.2
    )
    _run_solver_once(solver, optional=[obj])
    params = obj.get_params()
    params["weight"] = 0.4
    obj.update_params(params)
    _run_solver_once(solver, optional=[obj])


def test_sdf_self_collision_penalty(solver_with_fake_sdf):
    solver = solver_with_fake_sdf
    obj = SDFSelfCollisionPenaltyObj(
        bone_names=solver.controlled_bones,
        num_samples_per_bone=2,
        min_dist=0.0,
        weight=0.5,
    )
    _run_solver_once(solver, optional=[obj])
    p = obj.get_params()
    p["weight"] = 1.0
    obj.update_params(p)
    _run_solver_once(solver, optional=[obj])


def test_distance_obj_fk_accuracy_after_update(solver, model_path):
    """IK accuracy test with absolute targets on left_index3 using statics bounds.
    Uses original absolute targets; constructs solver including finger chain with official bounds from smplx_statics.
    """
    from jax_ik.ik import InverseKinematicsSolver
    from jax_ik.objectives import DistanceObjTraj

    old_target = np.array([0.0, 0.2, 0.3], dtype=np.float32)
    new_target = np.array([0.0, 0.0, 0.4], dtype=np.float32)

    base_bones = solver.controlled_bones
    finger_bones = ["left_index1", "left_index2", "left_index3"]
    extended_bones = base_bones + [b for b in finger_bones if b not in base_bones]

    # Build bounds from smplx_statics (degrees)
    bounds = []
    for bone in extended_bones:
        if bone not in left_arm_bounds_dict:
            pytest.skip(f"Bone {bone} lacks bounds in smplx_statics; cannot run test")
        lower, upper = left_arm_bounds_dict[bone]
        for l, u in zip(lower, upper):
            bounds.append((l, u))

    ext_solver = InverseKinematicsSolver(
        model_file=model_path,
        controlled_bones=extended_bones,
        bounds=bounds,
        threshold=1e-3,
        num_steps=10000,
        compute_sdf=False,
    )
    fk_solver = ext_solver.fk_solver
    end_bone = "left_index3"
    assert end_bone in fk_solver.bone_names

    init_angles = np.zeros(len(extended_bones) * 3, dtype=np.float32)

    fk0 = fk_solver.compute_fk_from_angles(init_angles)
    head0, tail0 = fk_solver.get_bone_head_tail_from_fk(fk0, end_bone)
    start_pos = np.asarray(tail0)
    start_dist_old = np.linalg.norm(start_pos - old_target)

    obj = DistanceObjTraj(
        bone_name=end_bone, target_points=[old_target], use_head=False, weight=1.0
    )
    traj1, loss1, steps1 = ext_solver.solve(
        initial_rotations=init_angles,
        learning_rate=0.25,
        mandatory_objective_functions=[obj],
        ik_points=2,
        patience=9000,
        verbose=False,
    )
    fk1 = fk_solver.compute_fk_from_angles(traj1[-1])
    _, tail1 = fk_solver.get_bone_head_tail_from_fk(fk1, end_bone)
    eff1 = np.asarray(tail1)
    dist1 = np.linalg.norm(eff1 - old_target)

    assert (dist1 < 0.07) or (dist1 < start_dist_old * 0.4), (
        f"Insufficient convergence to old target. start={start_dist_old:.3f} final={dist1:.3f} steps={steps1} loss={loss1:.4f})"
    )

    params = obj.get_params()
    params["target_points"] = [new_target.tolist()]
    obj.update_params(params)

    prev_to_new = np.linalg.norm(eff1 - new_target)

    traj2, loss2, steps2 = ext_solver.solve(
        initial_rotations=traj1[-1],
        learning_rate=0.25,
        mandatory_objective_functions=[obj],
        ik_points=2,
        patience=9000,
        verbose=False,
    )
    fk2 = fk_solver.compute_fk_from_angles(traj2[-1])
    _, tail2 = fk_solver.get_bone_head_tail_from_fk(fk2, end_bone)
    eff2 = np.asarray(tail2)
    dist2 = np.linalg.norm(eff2 - new_target)

    assert (dist2 < 0.07) or (dist2 < prev_to_new * 0.4), (
        f"Insufficient convergence to new target. prev={prev_to_new:.3f} final={dist2:.3f} steps={steps2} loss={loss2:.4f}"
    )
    assert dist2 <= prev_to_new + 1e-6, "No improvement toward new target after update"
    moved = np.linalg.norm(traj2[-1] - traj1[-1]) > 1e-6
    assert moved or dist2 < 1e-3, "Angles unchanged despite target update"
