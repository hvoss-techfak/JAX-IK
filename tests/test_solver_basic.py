import numpy as np

from jax_ik.hand_specification import HandSpecification
from jax_ik.objectives import (
    BoneDirectionObjective,
    BoneZeroRotationObj,
    DistanceObjTraj,
)


def test_basic_solve_runs(solver, zero_angles):
    # Single target near origin so small movement needed
    target = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    mand = [DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=1.0)]
    opt = [BoneZeroRotationObj(weight=0.01)]

    result, loss, steps = solver.solve(
        initial_rotations=zero_angles,
        learning_rate=0.15,
        mandatory_objective_functions=mand,
        optional_objective_functions=opt,
        ik_points=2,
        patience=30,
        verbose=False,
    )
    # result shape (T, D)
    assert result.ndim == 2
    assert result.shape[1] == zero_angles.shape[0]
    assert result.shape[0] == 3  # 1 initial + ik_points(2)
    assert steps > 0
    assert np.isfinite(loss)
    assert loss >= 0


def test_reuse_pool_and_param_update(solver, zero_angles):
    # Run twice with changed weight to ensure no retrace errors and pool update works
    target = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    mand1 = [DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=1.0)]
    res1, loss1, steps1 = solver.solve(zero_angles, mandatory_objective_functions=mand1, ik_points=1, verbose=False)
    # Update weight
    mand2 = [DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=0.5)]
    res2, loss2, steps2 = solver.solve(zero_angles, mandatory_objective_functions=mand2, ik_points=1, verbose=False)
    assert res1.shape == res2.shape
    # The loss with lower weight should not be greater than previous * 1.1 (loose check)
    assert loss2 <= loss1 * 1.1
    assert steps1 > 0 and steps2 > 0


def test_multiple_optional_objectives(solver, zero_angles):
    target = np.array([0.03, 0.01, 0.02], dtype=np.float32)
    mand = [DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=1.0)]
    # Add several zero-rotation penalties with different weights to test pool fill
    opt = [BoneZeroRotationObj(weight=w) for w in [0.01, 0.02, 0.03]]
    result, loss, _ = solver.solve(zero_angles, mandatory_objective_functions=mand, optional_objective_functions=opt, ik_points=1, verbose=False)
    assert result.shape[0] == 2
    assert np.isfinite(loss)


def test_objective_type_swap_pool(solver, zero_angles):
    # First run with DistanceObjTraj
    target = np.array([0.01, 0.02, 0.01], dtype=np.float32)
    mand1 = [DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=1.0)]
    solver.solve(initial_rotations=zero_angles, mandatory_objective_functions=mand1, ik_points=1, verbose=False)
    # Now replace mandatory objective with different type to force pool structural change
    mand2 = [BoneDirectionObjective(bone_name="left_elbow", directions=[[0,1,0]], weight=0.5)]
    solver.solve(initial_rotations=zero_angles, mandatory_objective_functions=mand2, ik_points=1, verbose=False)
    # If no exception thrown, pool replacement worked.


def test_hand_specification_generates_objectives_and_runs(solver, zero_angles):
    spec = HandSpecification(is_pointing=True, look_forward=True)
    objectives = spec.get_objectives(left_hand=True, controlled_bones=solver.controlled_bones, full_trajectory=False, last_position=True, weight=0.5)
    assert objectives, "HandSpecification should produce objectives"
    # Add Distance objective so solver moves wrist slightly
    target = np.array([0.04, 0.02, 0.03], dtype=np.float32)
    dist_obj = DistanceObjTraj(bone_name="left_wrist", target_points=[target], use_head=True, weight=1.0)
    # Run with combined objectives
    all_mand = [dist_obj] + objectives
    res, loss, steps = solver.solve(initial_rotations=zero_angles, mandatory_objective_functions=all_mand, ik_points=1, verbose=False)
    assert res.shape[0] == 2
    assert np.isfinite(loss)
