"""Per-algorithm solve wrappers for UR5 (evaluation 3).

jax_ik itself reuses algos.solve_jax_ik/reverify_success unchanged (via a
common.ModelSpec pointed at UR5_PATH/UR5_BONES/tool0, ee_use_head=True) --
that function only ever deals in jax_ik's own 18-DOF angle convention and
model_file/controlled_bones/ee_bone were already parameterized for
evaluation 2, so no UR5-specific code is needed there. ikpy and
roboticstoolbox get their own wrappers here since they operate on UR5's
native 6-DOF single-axis chain (see ur5_common.py) instead of jax_ik's
18-DOF representation.
"""

import time

import numpy as np

from . import ur5_common as ur5
from . import targets_ur5 as targets_mod


def solve_ikpy(chain, tier, pos_jax, R_jax=None):
    import scipy.optimize as spo

    pos_native, R_native = ur5.native_from_jax(pos_jax, R_jax)
    target = np.eye(4)
    target[:3, 3] = pos_native
    kwargs = {}
    if tier == "hard":
        target[:3, :3] = R_native
        kwargs["orientation_mode"] = "all"

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

    q6 = np.asarray(full_angles[1:-1], dtype=np.float32)
    iterations = int(captured.get("nfev", 0))
    return elapsed_ms, iterations, q6


def solve_rtb(robot, tier, pos_jax, R_jax=None):
    import spatialmath as sm

    pos_native, R_native = ur5.native_from_jax(pos_jax, R_jax)
    if tier == "hard":
        U, _, Vt = np.linalg.svd(np.asarray(R_native, dtype=np.float64))
        R64 = U @ Vt
        if np.linalg.det(R64) < 0:
            U[:, -1] *= -1
            R64 = U @ Vt
        T = sm.SE3.Rt(R64, np.asarray(pos_native, dtype=np.float64))
        mask = [1, 1, 1, 1, 1, 1]
    else:
        T = sm.SE3.Trans(*pos_native)
        mask = [1, 1, 1, 0, 0, 0]

    start = time.perf_counter()
    sol = robot.ikine_LM(T, end=ur5.EE_LINK, mask=mask, joint_limits=True, ilimit=30, slimit=100, tol=1e-6)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    q6 = np.asarray(sol.q, dtype=np.float32)
    return elapsed_ms, int(sol.iterations), q6


def reverify_success(fk_solver, tier, q6, pos_jax, R_jax=None):
    from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

    angles18 = ur5.angle18_from_q6(q6)
    pos_obj = DistanceObjTraj(bone_name=ur5.EE_LINK, target_points=pos_jax, use_head=True, weight=1.0)
    loss = float(pos_obj(angles18, fk_solver))
    if tier == "hard":
        ori_obj = EndEffectorOrientationObj(
            bone_name=ur5.EE_LINK, target_transform=targets_mod._r_to_4x4(R_jax), weight=1.0
        )
        loss += float(ori_obj(angles18, fk_solver))
    return loss < targets_mod.TIER_THRESHOLD[tier]
