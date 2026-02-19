import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jax_ik.ik import InverseKinematicsSolver
from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

# Optional baseline solvers (only used when explicitly requested).
try:
    import ikpy.chain  # type: ignore
except Exception:  # pragma: no cover
    print("Warning: IKPy not available. IKPy results will be skipped.")
    ikpy = None  # type: ignore

try:
    from trac_ik import TracIK  # type: ignore
except Exception:  # pragma: no cover
    print("Warning: TRAC-IK not available. TRAC-IK results will be skipped.")
    TracIK = None  # type: ignore

try:
    from ur_ikfast import ur_kinematics  # type: ignore
except Exception:  # pragma: no cover
    print("Warning: IKFast UR kinematics not available. IKFast results will be skipped.")
    ur_kinematics = None  # type: ignore


@dataclass
class Stats:
    success: int = 0
    elapsed: float = 0.0
    failure_distance_sum: float = 0.0
    failure_angle_sum_deg: float = 0.0


def _rotation_angle_deg(R_target: np.ndarray, R_result: np.ndarray) -> float:
    """Return the geodesic angle between two rotation matrices in degrees.

    Uses a quaternion-based formula for numerical stability.
    """

    def _quat_from_R(R: np.ndarray) -> np.ndarray:
        # Robust conversion using the (trace+1) branch; clip to avoid negative due to drift.
        tr = float(np.trace(R))
        w2 = max(0.0, (tr + 1.0) * 0.25)
        w = float(np.sqrt(w2))
        if w < 1e-9:
            # Fallback: use diagonal-based selection (rare for ~180deg)
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                x = np.sqrt(max(0.0, (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 0.25))
                y = (R[0, 1] + R[1, 0]) / (4.0 * x + 1e-12)
                z = (R[0, 2] + R[2, 0]) / (4.0 * x + 1e-12)
                w = (R[2, 1] - R[1, 2]) / (4.0 * x + 1e-12)
                q = np.array([w, x, y, z], dtype=np.float64)
            elif i == 1:
                y = np.sqrt(max(0.0, (1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 0.25))
                x = (R[0, 1] + R[1, 0]) / (4.0 * y + 1e-12)
                z = (R[1, 2] + R[2, 1]) / (4.0 * y + 1e-12)
                w = (R[0, 2] - R[2, 0]) / (4.0 * y + 1e-12)
                q = np.array([w, x, y, z], dtype=np.float64)
            else:
                z = np.sqrt(max(0.0, (1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 0.25))
                x = (R[0, 2] + R[2, 0]) / (4.0 * z + 1e-12)
                y = (R[1, 2] + R[2, 1]) / (4.0 * z + 1e-12)
                w = (R[1, 0] - R[0, 1]) / (4.0 * z + 1e-12)
                q = np.array([w, x, y, z], dtype=np.float64)
        else:
            x = (R[2, 1] - R[1, 2]) / (4.0 * w + 1e-12)
            y = (R[0, 2] - R[2, 0]) / (4.0 * w + 1e-12)
            z = (R[1, 0] - R[0, 1]) / (4.0 * w + 1e-12)
            q = np.array([w, x, y, z], dtype=np.float64)

        q /= (np.linalg.norm(q) + 1e-12)
        return q

    Rt = np.asarray(R_target[:3, :3], dtype=np.float64)
    Rr = np.asarray(R_result[:3, :3], dtype=np.float64)

    # Relative rotation
    R_rel = Rt.T @ Rr
    q = _quat_from_R(R_rel)

    # For unit quaternion, angle = 2*acos(|w|)
    w = float(np.clip(abs(q[0]), 0.0, 1.0))
    ang = float(np.degrees(2.0 * np.arccos(w)))
    if not np.isfinite(ang):
        return float("inf")
    return ang


def _fk_tip_pose(solver: InverseKinematicsSolver, angles: np.ndarray, tip_bone: str):
    fk = solver.fk_solver.compute_fk_from_angles(angles)
    idx = solver.fk_solver.bone_names.index(tip_bone)
    T = np.asarray(fk[idx])
    pos = np.asarray(T[:3, 3])
    return pos, T


def _infer_tip_bone(solver: InverseKinematicsSolver) -> str:
    # Prefer common UR end-effector link names if present.
    candidates = [
        "tool0",
        "ee_link",
        "end_effector",
        "wrist_3_link",
    ]
    for c in candidates:
        if c in solver.fk_solver.bone_names:
            return c

    # Fallback: choose any leaf bone (no children) and pick the last one for stability.
    leaves = [
        name
        for name in solver.fk_solver.bone_names
        if not solver.fk_solver.skeleton.get(name, {}).get("children")
    ]
    if leaves:
        return leaves[-1]

    return solver.fk_solver.bone_names[-1]


def _ikpy_fk(chain, q_active: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pos, T) using IKPy's FK for a vector of active joint angles."""
    if chain is None:
        return None

    q_active = np.asarray(q_active, dtype=np.float64).reshape(-1)

    # IKPy uses one value per link; fixed links still occupy a slot.
    q_full = np.zeros(len(chain.links), dtype=np.float64)
    j = 0
    for i, link in enumerate(chain.links):
        if getattr(link, "joint_type", None) != "fixed":
            if j < q_active.size:
                q_full[i] = float(q_active[j])
            j += 1

    T = np.asarray(chain.forward_kinematics(q_full), dtype=np.float64)
    pos = np.asarray(T[:3, 3], dtype=np.float64)
    return pos, T


def _ikfast_fk(kin, q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pos, T) using IKFast's forward() output."""
    if kin is None:
        return None
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    t = np.asarray(kin.forward(q.tolist()), dtype=np.float64).reshape(-1)

    # IKFast pose is [x,y,z,qx,qy,qz,qw]
    pos = t[:3]
    qx, qy, qz, qw = t[3], t[4], t[5], t[6]

    # quaternion -> rotation matrix
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w
    if n < 1e-16:
        R = np.eye(3, dtype=np.float64)
    else:
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        R = np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float64,
        )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return pos, T


def _tracik_fk(tracik_solver, q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if tracik_solver is None:
        return None
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    pos, T = tracik_solver.fk(q)
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    T = np.asarray(T, dtype=np.float64)
    return pos, T


def _build_bounds_for_controlled_bones(
    solver: InverseKinematicsSolver,
) -> list[tuple[float, float]]:
    """Return radian bounds for the solver angle vector (already expanded to 3 per bone)."""
    return list(zip(np.asarray(solver.lower_bounds), np.asarray(solver.upper_bounds)))


def _pose_T_from_pos_R(target_pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R[:3, :3], dtype=np.float64)
    T[:3, 3] = np.asarray(target_pos[:3], dtype=np.float64)
    return T


def _ikpy_solve(
    urdf_path: str,
    active_joint_count: int,
    target_pos: np.ndarray,
    target_T: np.ndarray,
    seed_q: np.ndarray,
) -> np.ndarray | None:
    if ikpy is None:
        return None

    chain = ikpy.chain.Chain.from_urdf_file(urdf_path)
    # IKPy expects a full joint vector including fixed joints; simplest is to provide an initial guess of zeros.
    q0 = np.zeros(len(chain.links), dtype=np.float64)
    # Map our active joints into the chain's active joints order: IKPy uses one angle per link.
    # We assume revolute joints correspond sequentially and ignore fixed links; this matches typical URDFs.
    # Seed is used for stability if sizes match.
    if seed_q is not None:
        j = 0
        for i, link in enumerate(chain.links):
            if getattr(link, "joint_type", None) != "fixed":
                if j < len(seed_q):
                    q0[i] = float(seed_q[j])
                j += 1

    target_frame = np.asarray(target_T, dtype=np.float64)
    # Newer IKPy versions support full 4x4 targets via `target_matrix`.
    try:
        sol = chain.inverse_kinematics(target_matrix=target_frame, initial_position=q0)
    except TypeError:
        # Fallback: position-only IK.
        sol = chain.inverse_kinematics(target_pos, initial_position=q0)

    # Extract the first N active joints.
    out = []
    for i, link in enumerate(chain.links):
        if getattr(link, "joint_type", None) != "fixed":
            out.append(float(sol[i]))
            if len(out) >= active_joint_count:
                break
    if len(out) != active_joint_count:
        return None
    return np.asarray(out, dtype=np.float32)


def _ikfast_solve(
    robot_name: str,
    target_pos: np.ndarray,
    target_T: np.ndarray,
    seed_q: np.ndarray,
) -> np.ndarray | None:
    if ur_kinematics is None:
        return None
    if robot_name.lower() not in {"ur5", "ur5e", "ur3", "ur10", "ur10e", "ur3e"}:
        return None

    kin = ur_kinematics.URKinematics(robot_name.lower())
    # IKFast wrapper typically uses a 7D pose: [x,y,z,qx,qy,qz,qw]
    # Compute quaternion from rotation matrix.
    R = np.asarray(target_T[:3, :3], dtype=np.float64)
    # Robust quaternion from rotation matrix.
    tr = float(np.trace(R))
    qw = np.sqrt(max(0.0, 1.0 + tr)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-12)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-12)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-12)

    t = np.array(
        [
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            float(qx),
            float(qy),
            float(qz),
            float(qw),
        ],
        dtype=np.float64,
    )

    zeros = np.zeros(6, dtype=np.float64)
    start = time.perf_counter()
    sol = kin.inverse(t, False, zeros)
    _ = time.perf_counter() - start

    if sol is None:
        return None
    return np.asarray(sol, dtype=np.float32)


def _tracik_solve(
    urdf_path: str,
    base_link_name: str,
    tip_link_name: str,
    target_pos: np.ndarray,
    target_T: np.ndarray,
    seed_q: np.ndarray,
) -> np.ndarray | None:
    if TracIK is None:
        return None
    solver = TracIK(
        urdf_path=urdf_path,
        base_link_name=base_link_name,
        tip_link_name=tip_link_name,
    )
    # TracIK expects position and rotation matrix.
    t_p = np.asarray(target_pos, dtype=np.float64)
    t_r = np.asarray(target_T[:3, :3], dtype=np.float64)
    q0 = np.asarray(seed_q[: len(solver.joint_limits[0])], dtype=np.float64)
    sol = solver.ik(t_p, t_r, q0)
    if sol is None:
        return None
    return np.asarray(sol, dtype=np.float32)


def _repo_root() -> Path:
    """Best-effort repository root (directory containing pyproject.toml).

    This keeps CLI defaults and CI stable even if the script is invoked from a
    different working directory.
    """

    cur = Path(__file__).resolve()
    for p in (cur.parent, *cur.parents):
        if (p / "pyproject.toml").is_file():
            return p
    # Fallback: assume repo root is one level above this file.
    return cur.parent.parent


def _resolve_urdf_path(urdf_arg: str) -> Path:
    """Resolve a URDF path robustly.

    Resolution order for relative paths:
      1) current working directory
      2) repo root (pyproject.toml)
      3) directory containing this script
    """

    raw = Path(urdf_arg).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [
        (Path.cwd() / raw),
        (_repo_root() / raw),
        (Path(__file__).resolve().parent / raw),
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Special-case: allow passing just a robot stem like "UR5".
    stem = raw.as_posix()
    stem = stem[:-5] if stem.lower().endswith(".urdf") else stem
    candidates += [
        (_repo_root() / "models" / f"{stem}.urdf"),
        (_repo_root() / "models" / f"{stem.upper()}.urdf"),
        (_repo_root() / "models" / f"{stem.lower()}.urdf"),
    ]
    for c in candidates[-3:]:
        if c.is_file():
            return c

    msg = "URDF path not found. Tried:\n" + "\n".join(f"- {c}" for c in candidates)
    raise ValueError(msg)


def _as_fk_angle_vector(
    solver: InverseKinematicsSolver,
    q: np.ndarray,
) -> np.ndarray:
    """Coerce an input vector into the FK solver's expected angle vector shape.

    JAX-IK's FK expects a flat vector of Euler XYZ angles for each controlled bone:
        shape == (3 * len(controlled_bones),)

    Some baseline IK solvers (IKPy / IKFast / TRAC-IK) return one value per joint
    (typically 6 for UR robots). For evaluation, we expand that (N,) vector into
    per-bone Euler angles by mapping each scalar to a Z-rotation: [0, 0, q_i].

    This doesn't change JAX-IK itself; it only prevents the evaluation harness
    from crashing when it tries to run FK on baseline solutions.
    """

    q = np.asarray(q, dtype=np.float32).reshape(-1)

    expected = int(len(solver.fk_solver.controlled_indices) * 3)
    if q.size == expected:
        return q

    # Baseline solvers often return one angle per joint (1-DOF each).
    if expected % 3 == 0 and q.size == expected // 3:
        out = np.zeros((q.size, 3), dtype=np.float32)
        out[:, 2] = q
        return out.reshape(-1)

    raise ValueError(
        f"Unexpected joint vector size for FK: got {q.size}, expected {expected} "
        f"(or {expected//3} for 1-DOF baselines)."
    )


def run(
    tests: int,
    seed: int,
    urdf_path: str,
    tip_bone: str | None,
    threshold: float,
    num_steps: int,
    learning_rate: float,
    patience: int,
    distance_error: float,
    angle_error_deg: float,
    warmup: int,
    output_csv: str,
    compute_sdf: bool,
    orientation_weight: float,
    restarts: int,
    lr_decay: float,
    methods: list[str],
    robot_name: str,
    tracik_base_link: str,
    tracik_tip_link: str | None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    urdf_p = _resolve_urdf_path(urdf_path)
    urdf_path = str(urdf_p.resolve())

    # Pre-build baseline solver instances/chains so we can compute FK consistently.
    ikpy_chain = None
    if "ikpy" in methods and ikpy is not None:
        ikpy_chain = ikpy.chain.Chain.from_urdf_file(urdf_path)

    ikfast_kin = None
    if "ikfast" in methods and ur_kinematics is not None:
        try:
            ikfast_kin = ur_kinematics.URKinematics(robot_name.lower())
        except Exception:
            ikfast_kin = None

    tracik_solver = None
    if "trac-ik" in methods and TracIK is not None:
        tracik_solver = TracIK(
            urdf_path=urdf_path,
            base_link_name=tracik_base_link,
            tip_link_name=tracik_tip_link or "tool0",
        )

    # Build solver. If we don't specify controlled bones, there are none.
    # For UR5 we want all movable links (those that have limits) as controlled.
    tmp_solver = InverseKinematicsSolver(
        model_file=urdf_path,
        controlled_bones=[],
        bounds=[(-180, 180)] * 6 * 3,
        threshold=threshold,
        num_steps=num_steps,
        compute_sdf=compute_sdf,
    )
    movable = [
        b
        for b in tmp_solver.fk_solver.limits.keys()
        if b in tmp_solver.fk_solver.bone_names
    ]

    # Many URDFs expose limits for extra fixed/virtual links (e.g. base/tool/ee).
    # Baseline IK solvers (IKPy/IKFast/TRAC-IK) typically return only the actuated
    # revolute joints (e.g. 6 for UR5). To keep evaluation consistent across
    # solvers, default to the standard arm joint links when available.
    preferred = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    if all(p in movable for p in preferred):
        movable = preferred

    solver = InverseKinematicsSolver(
        model_file=urdf_path,
        controlled_bones=movable,
        bounds=None,
        threshold=threshold,
        num_steps=num_steps,
        compute_sdf=compute_sdf,
    )

    if tip_bone is None:
        tip_bone = _infer_tip_bone(solver)

    if tracik_tip_link is None:
        tracik_tip_link = tip_bone

    bounds_rad = _build_bounds_for_controlled_bones(solver)

    # Warmup (JAX compilation) excluded from stats
    if warmup > 0 and "jax-ik" in methods:
        q0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds_rad], dtype=np.float32)
        target_pos, target_T = _fk_tip_pose(solver, q0, tip_bone)
        mand = (
            DistanceObjTraj(
                bone_name=tip_bone,
                target_points=np.asarray(target_pos, dtype=np.float32),
                use_head=True,
                weight=1.0,
            ),
            EndEffectorOrientationObj(
                bone_name=tip_bone,
                target_transform=target_T,
                weight=orientation_weight,
            ),
        )
        for _ in range(warmup):
            solver.solve(
                initial_rotations=q0,
                learning_rate=learning_rate,
                mandatory_objective_functions=mand,
                optional_objective_functions=(),
                ik_points=1,
                patience=patience,
                verbose=False,
            )

    stats_by = {m: Stats() for m in methods}

    active_joint_count = int(len(solver.fk_solver.controlled_indices))

    for _ in range(tests):
        # Random target in bounds.
        q = np.array(
            [random.uniform(lo, hi) for lo, hi in bounds_rad], dtype=np.float32
        )

        target_pos, target_T = _fk_tip_pose(solver, q, tip_bone)

        mand = (
            DistanceObjTraj(
                bone_name=tip_bone,
                target_points=np.asarray(target_pos, dtype=np.float32),
                use_head=True,
                weight=1.0,
            ),
            EndEffectorOrientationObj(
                bone_name=tip_bone,
                target_transform=target_T,
                weight=orientation_weight,
            ),
        )

        # JAX-IK (multi-start)
        if "jax-ik" in methods:
            best_elapsed = 0.0
            best_dist = float("inf")
            best_ang = float("inf")
            for r in range(max(1, restarts)):
                init = q if r == 0 else np.array(
                    [random.uniform(lo, hi) for lo, hi in bounds_rad], dtype=np.float32
                )
                lr = float(learning_rate) * (float(lr_decay) ** r)
                start = time.perf_counter()
                solved_traj, _, _ = solver.solve(
                    initial_rotations=init,
                    learning_rate=lr,
                    mandatory_objective_functions=mand,
                    optional_objective_functions=(),
                    ik_points=1,
                    patience=patience,
                    verbose=False,
                )
                elapsed = time.perf_counter() - start
                solved = np.asarray(solved_traj[-1], dtype=np.float32)
                result_pos, result_T = _fk_tip_pose(solver, solved, tip_bone)
                dist = float(np.linalg.norm(result_pos - target_pos))
                ang = _rotation_angle_deg(target_T, result_T)

                score = dist + np.deg2rad(ang)
                best_score = best_dist + np.deg2rad(best_ang)
                if score < best_score:
                    best_dist, best_ang, best_elapsed = dist, ang, elapsed
                if best_dist <= distance_error and best_ang <= angle_error_deg:
                    break

            s = stats_by["jax-ik"]
            s.elapsed += best_elapsed
            if best_dist <= distance_error and best_ang <= angle_error_deg:
                s.success += 1
            else:
                s.failure_distance_sum += best_dist
                s.failure_angle_sum_deg += best_ang

        # IKPy
        if "ikpy" in methods:
            start = time.perf_counter()
            sol = _ikpy_solve(urdf_path, active_joint_count, target_pos, target_T, q)
            elapsed = time.perf_counter() - start
            if sol is None or ikpy_chain is None:
                dist, ang = float("inf"), float("inf")
            else:
                fk_res = _ikpy_fk(ikpy_chain, sol)
                if fk_res is None:
                    dist, ang = float("inf"), float("inf")
                else:
                    result_pos, result_T = fk_res
                    dist = float(np.linalg.norm(result_pos - target_pos))
                    ang = _rotation_angle_deg(target_T, result_T)
            s = stats_by["ikpy"]
            s.elapsed += elapsed
            if dist <= distance_error and ang <= angle_error_deg:
                s.success += 1
            else:
                s.failure_distance_sum += dist
                s.failure_angle_sum_deg += ang

        # IKFast
        if "ikfast" in methods:
            start = time.perf_counter()
            sol = _ikfast_solve(robot_name, target_pos, target_T, q)
            elapsed = time.perf_counter() - start
            if sol is None or ikfast_kin is None:
                dist, ang = float("inf"), float("inf")
            else:
                fk_res = _ikfast_fk(ikfast_kin, sol)
                if fk_res is None:
                    dist, ang = float("inf"), float("inf")
                else:
                    result_pos, result_T = fk_res
                    dist = float(np.linalg.norm(result_pos - target_pos))
                    ang = _rotation_angle_deg(target_T, result_T)
            s = stats_by["ikfast"]
            s.elapsed += elapsed
            if dist <= distance_error and ang <= angle_error_deg:
                s.success += 1
            else:
                s.failure_distance_sum += dist
                s.failure_angle_sum_deg += ang

        # TRAC-IK
        if "trac-ik" in methods:
            start = time.perf_counter()
            sol = _tracik_solve(
                urdf_path,
                tracik_base_link,
                tracik_tip_link,
                target_pos,
                target_T,
                q,
            )
            elapsed = time.perf_counter() - start
            if sol is None or tracik_solver is None:
                dist, ang = float("inf"), float("inf")
            else:
                fk_res = _tracik_fk(tracik_solver, sol)
                if fk_res is None:
                    dist, ang = float("inf"), float("inf")
                else:
                    result_pos, result_T = fk_res
                    dist = float(np.linalg.norm(result_pos - target_pos))
                    ang = _rotation_angle_deg(target_T, result_T)
            s = stats_by["trac-ik"]
            s.elapsed += elapsed
            if dist <= distance_error and ang <= angle_error_deg:
                s.success += 1
            else:
                s.failure_distance_sum += dist
                s.failure_angle_sum_deg += ang

    header = (
        "Solver,Success Rate (%),Average Elapsed Time (s),Average Failure Distance (mm),Average Failure Angle (°)"
    )

    lines = [header]
    for m in methods:
        st = stats_by[m]
        failures = tests - st.success
        if failures > 0:
            avg_fail_dist_mm = (st.failure_distance_sum / failures) * 1000.0
            avg_fail_ang = st.failure_angle_sum_deg / failures
            if not np.isfinite(avg_fail_dist_mm):
                avg_fail_dist_mm = float("inf")
            if not np.isfinite(avg_fail_ang):
                avg_fail_ang = float("inf")
            fail_dist_str = f"{avg_fail_dist_mm} mm"
            fail_ang_str = f"{avg_fail_ang} °"
        else:
            fail_dist_str = "-"
            fail_ang_str = "-"

        line = (
            f"{m},{st.success / tests * 100}%,{(st.elapsed / tests)*1000} ms,{fail_dist_str},{fail_ang_str}"
        )
        lines.append(line)

    s = "\n".join(lines)
    print(s)
    with open(output_csv, "w", encoding="utf-8", errors="ignore") as f:
        f.write(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tests", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--urdf",
        type=str,
        # Default to the repo's canonical UR5 URDF.
        default=str(_repo_root() / "models" / "UR5.urdf"),
    )
    p.add_argument("--tip-bone", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--distance-error", type=float, default=0.01)
    p.add_argument("--angle-error-deg", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output-csv", type=str, default="Additional.csv")
    p.add_argument(
        "--compute-sdf", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--orientation-weight", type=float, default=1.0)
    p.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="Number of random restarts per test case (higher = higher success, slower).",
    )
    p.add_argument(
        "--lr-decay",
        type=float,
        default=0.6,
        help="Multiplier applied to learning-rate for each successive restart.",
    )

    p.add_argument(
        "--methods",
        type=str,
        default="jax-ik,ikpy,ikfast,trac-ik",
        help="Comma-separated list of solvers to run: jax-ik,ikpy,ikfast,trac-ik",
    )
    p.add_argument(
        "--robot-name",
        type=str,
        default="ur5",
        help="Robot name for IKFast backend (e.g. ur5).",
    )
    p.add_argument(
        "--tracik-base-link",
        type=str,
        default="base_link",
        help="Base link name for TRAC-IK.",
    )
    p.add_argument(
        "--tracik-tip-link",
        type=str,
        default=None,
        help="Tip link name for TRAC-IK (defaults to --tip-bone).",
    )

    args = p.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    # Normalize a couple common spellings.
    methods = ["trac-ik" if m in {"tracik", "trac_ik", "trac-ik"} else m for m in methods]
    methods = ["jax-ik" if m in {"jax", "jaxik", "jax-ik"} else m for m in methods]

    run(
        tests=args.tests,
        seed=args.seed,
        urdf_path=args.urdf,
        tip_bone=args.tip_bone,
        threshold=args.threshold,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        patience=args.patience,
        distance_error=args.distance_error,
        angle_error_deg=args.angle_error_deg,
        warmup=args.warmup,
        output_csv=args.output_csv,
        compute_sdf=args.compute_sdf,
        orientation_weight=args.orientation_weight,
        restarts=args.restarts,
        lr_decay=args.lr_decay,
        methods=methods,
        robot_name=args.robot_name,
        tracik_base_link=args.tracik_base_link,
        tracik_tip_link=args.tracik_tip_link,
    )


if __name__ == "__main__":
    main()

