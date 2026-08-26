"""Target generation + reachability verification for UR5 (evaluation 3).

Same two-tier (easy/strict) ladder and verify-by-actually-solving
methodology as targets.py, but working in UR5's native 6-DOF single-axis
joint space (q6) instead of the SMPL-X 3-DOF-per-bone space -- see
ur5_common.py's module docstring for why. Targets are stored in jax_ik's
own world frame (position + orientation of "tool0"); ikpy/roboticstoolbox
callers convert to the native URDF frame themselves via
ur5_common.native_from_jax right before solving.
"""

import json
import os

import numpy as np

from . import ur5_common as ur5

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

NUM_TARGETS = 100
SEED_BASE = 20260827

POSITION_THRESHOLD = 0.005
HARD_THRESHOLD = 0.01

TIER_THRESHOLD = {"easy": POSITION_THRESHOLD, "medium": POSITION_THRESHOLD, "hard": HARD_THRESHOLD}


def tier_bounds_deg(tier):
    if tier == "easy":
        return [(-180.0, 180.0)] * 6
    return [(-ur5.STRICT_DEG, ur5.STRICT_DEG)] * 6


def _cache_path(tier):
    return os.path.join(_PKG_DIR, f"compare_targets_ur5_{tier}.json")


def _r_to_4x4(R):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T


def _verify_reachable(jax_solver, tier, pos, R):
    from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

    mandatory = [
        DistanceObjTraj(bone_name=ur5.EE_LINK, target_points=pos, use_head=True, weight=1.0)
    ]
    if tier == "hard":
        mandatory.append(
            EndEffectorOrientationObj(bone_name=ur5.EE_LINK, target_transform=_r_to_4x4(R), weight=1.0)
        )
    _, best_obj, _ = jax_solver.solve(
        initial_rotations=None,
        learning_rate=0.2,
        mandatory_objective_functions=tuple(mandatory),
        optional_objective_functions=(),
        ik_points=1,
        patience=200,
        verbose=False,
    )
    return float(best_obj) < TIER_THRESHOLD[tier]


def generate_or_load(tier, n=NUM_TARGETS, verbose=True):
    path = _cache_path(tier)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("n") == n and len(data.get("targets", [])) == n:
            return data["targets"]

    bounds6 = tier_bounds_deg(tier)
    fk_solver = ur5.FKSolver(model_file=ur5.UR5_PATH, controlled_bones=ur5.UR5_BONES, do_compute_sdf=False)
    jax_solver = ur5.build_jax_ik_solver(bounds6, threshold=TIER_THRESHOLD[tier])

    lo = np.radians(np.asarray([b[0] for b in bounds6], dtype=np.float64))
    hi = np.radians(np.asarray([b[1] for b in bounds6], dtype=np.float64))
    tier_offset = {"easy": 0, "medium": 1, "hard": 2}[tier]
    rng = np.random.default_rng(SEED_BASE + tier_offset)

    targets = []
    attempts = 0
    while len(targets) < n:
        attempts += 1
        q6 = rng.uniform(lo, hi).astype(np.float32)
        angles18 = ur5.angle18_from_q6(q6)
        fk = fk_solver.compute_fk_from_angles(angles18)
        idx = fk_solver.bone_names.index(ur5.EE_LINK)
        T = np.asarray(fk[idx])
        pos = T[:3, 3].astype(np.float32)
        R = T[:3, :3].astype(np.float32)
        if _verify_reachable(jax_solver, tier, pos, R):
            entry = {"pos": pos.tolist()}
            if tier == "hard":
                entry["R"] = R.tolist()
            targets.append(entry)
            if verbose and len(targets) % 20 == 0:
                print(f"[ur5/{tier}] {len(targets)}/{n} verified (attempts={attempts})")
        if attempts > n * 20:
            raise RuntimeError(f"[ur5/{tier}] too many failed reachability attempts ({attempts})")

    with open(path, "w") as f:
        json.dump({"n": n, "seed": SEED_BASE, "targets": targets}, f)
    if verbose:
        print(f"[ur5/{tier}] generated+verified {n} targets in {attempts} attempts -> {path}")
    return targets


if __name__ == "__main__":
    for tier in ["easy", "medium", "hard"]:
        generate_or_load(tier)
