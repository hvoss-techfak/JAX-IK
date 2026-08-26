"""Per-algorithm solve wrappers for jax_ik, ikpy, and roboticstoolbox.

Each solve_* function takes one target (position, and for the hard tier a
3x3 orientation matrix) plus tier bounds, and returns
(elapsed_ms, iterations, final_angles_xyz_per_bone_radians) where the
returned angles are always in jax_ik's own [x,y,z]-per-bone convention (so
callers can re-verify success uniformly through jax_ik's FK regardless of
which library produced the angles).

Generalized (evaluation 2) to take a common.ModelSpec (bone name, use_head)
so the 7-bone finger chain reuses this unchanged; defaults to
common.ARM_SPEC so evaluation 1's already-executed call sites keep working.
UR5 (evaluation 3) does NOT use this module -- see ur5_common.py/algos_ur5.py.
"""

import time

import numpy as np

from . import common
from . import targets as targets_mod


def _threshold_for(tier):
    return targets_mod.TIER_THRESHOLD[tier]


# ---------------------------------------------------------------------------
# jax_ik
# ---------------------------------------------------------------------------
def solve_jax_ik(jax_solver, tier, pos, R=None, spec=common.ARM_SPEC):
    from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

    mandatory = [
        DistanceObjTraj(
            bone_name=spec.ee_bone, target_points=pos, use_head=spec.ee_use_head, weight=1.0
        )
    ]
    if tier == "hard":
        mandatory.append(
            EndEffectorOrientationObj(
                bone_name=spec.ee_bone, target_transform=targets_mod._r_to_4x4(R), weight=1.0
            )
        )
    start = time.perf_counter()
    angles, best_obj, steps = jax_solver.solve(
        initial_rotations=None,
        learning_rate=0.2,
        mandatory_objective_functions=tuple(mandatory),
        optional_objective_functions=(),
        ik_points=1,
        patience=200,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, int(steps), np.asarray(angles, dtype=np.float32)


# ---------------------------------------------------------------------------
# ikpy
# ---------------------------------------------------------------------------
def solve_ikpy(chain, tier, pos, R=None, spec=common.ARM_SPEC):
    import scipy.optimize as spo

    target = np.eye(4)
    target[:3, 3] = pos
    kwargs = {}
    if tier == "hard":
        target[:3, :3] = R
        kwargs["orientation_mode"] = "all"

    # ikpy's public inverse_kinematics_frame doesn't return an iteration
    # count -- capture it by wrapping the exact scipy.optimize.least_squares
    # call it makes internally (see ikpy.inverse_kinematics.
    # inverse_kinematic_optimization) and reading OptimizeResult.nfev
    # (function evaluations, not a distinct "outer loop" count like the
    # other algorithms report).
    captured = {}
    real_least_squares = spo.least_squares

    def _wrapped(*args, **kw):
        res = real_least_squares(*args, **kw)
        captured["nfev"] = res.nfev
        return res

    spo.least_squares = _wrapped
    try:
        start = time.perf_counter()
        full_angles = chain.inverse_kinematics_frame(target, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        spo.least_squares = real_least_squares

    # full_angles is in chain-link order (OriginLink, then per-bone Z,Y,X,
    # then the fixed tail link) -- strip origin/tail and reorder Z,Y,X ->
    # X,Y,Z per bone to match jax_ik's convention.
    active = np.asarray(full_angles[1:-1], dtype=np.float32)
    angles_xyz = common.zyx_to_xyz(active)
    iterations = int(captured.get("nfev", 0))
    return elapsed_ms, iterations, angles_xyz


# ---------------------------------------------------------------------------
# roboticstoolbox
# ---------------------------------------------------------------------------
def solve_rtb(robot, tier, pos, R=None, joint_limits=True, spec=common.ARM_SPEC):
    import spatialmath as sm

    if tier == "hard":
        # R comes from float32 jax FK; spatialmath's strict SO(3) check can
        # reject it on tiny orthonormality drift, so re-orthonormalize via
        # SVD (nearest rotation matrix) before handing it off.
        U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
        R64 = U @ Vt
        if np.linalg.det(R64) < 0:
            U[:, -1] *= -1
            R64 = U @ Vt
        T = sm.SE3.Rt(R64, np.asarray(pos, dtype=np.float64))
        mask = [1, 1, 1, 1, 1, 1]
    else:
        T = sm.SE3.Trans(*pos)
        mask = [1, 1, 1, 0, 0, 0]

    start = time.perf_counter()
    sol = robot.ikine_LM(T, mask=mask, joint_limits=joint_limits, ilimit=30, slimit=100, tol=1e-6)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    angles_zyx = np.asarray(sol.q, dtype=np.float32)
    angles_xyz = common.zyx_to_xyz(angles_zyx)
    return elapsed_ms, int(sol.iterations), angles_xyz


# ---------------------------------------------------------------------------
# Uniform success re-check: run any algorithm's returned angles (in jax_ik's
# [x,y,z]-per-bone convention) back through jax_ik's own FK/objective loss,
# so "success" means the same thing for every algorithm instead of trusting
# each library's own internal convergence flag.
# ---------------------------------------------------------------------------
def reverify_success(fk_solver, tier, angles_xyz, pos, R=None, spec=common.ARM_SPEC):
    from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

    pos_obj = DistanceObjTraj(
        bone_name=spec.ee_bone, target_points=pos, use_head=spec.ee_use_head, weight=1.0
    )
    loss = float(pos_obj(np.asarray(angles_xyz, dtype=np.float32), fk_solver))
    if tier == "hard":
        ori_obj = EndEffectorOrientationObj(
            bone_name=spec.ee_bone, target_transform=targets_mod._r_to_4x4(R), weight=1.0
        )
        loss += float(ori_obj(np.asarray(angles_xyz, dtype=np.float32), fk_solver))
    return loss < _threshold_for(tier)
