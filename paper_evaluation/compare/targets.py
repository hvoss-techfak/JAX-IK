"""Per-tier target generation + reachability verification.

Three tiers, matching the comparison's difficulty ladder:
  - easy:   position-only objective, loose (near-unconstrained) bounds.
  - medium: position-only objective, strict anatomical bounds.
  - hard:   position + orientation objective, strict anatomical bounds.

Targets are FK-sampled (guaranteed reachable in principle, exactly like
ik_jax_lib_bench.py's _generate_targets), then *actually verified* by
running jax_ik's own solver under that tier's bounds/objective and checking
it converges under threshold -- any sample that doesn't is discarded and
resampled. Cached to JSON next to this file so repeated runs use the same
100 points per tier.

Generalized (evaluation 2) to take a common.ModelSpec so the 7-bone finger
chain can reuse this unchanged; defaults to common.ARM_SPEC (evaluation 1,
already run) so its cache filenames (compare_targets_{tier}.json, no model
suffix) and behavior are untouched. A different spec.key gets its own
compare_targets_{key}_{tier}.json so the two never collide.
"""

import json
import os

import numpy as np

from . import common

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

NUM_TARGETS = 100
SEED_BASE = 20260826

# Position-only success gate: mean squared error over the 3 position
# components (matches ik_jax_lib_bench.py's THRESHOLD, same convention).
POSITION_THRESHOLD = 0.005

# Hard tier gates on position MSE + orientation angle^2 (radians^2) summed,
# each objective weighted 1.0 -- i.e. a combined budget of ~0.005 position
# + ~0.005 orientation, added together into one threshold.
HARD_THRESHOLD = 0.01

TIER_THRESHOLD = {
    "easy": POSITION_THRESHOLD,
    "medium": POSITION_THRESHOLD,
    "hard": HARD_THRESHOLD,
}


def tier_bounds_deg(tier, spec=common.ARM_SPEC):
    if tier == "easy":
        return common.loose_bounds_deg(spec.controlled_bones)
    return common.strict_bounds_deg(spec.controlled_bones)


# Kept for backward compat with eval-1 call sites (dict of no-arg callables
# over the default ARM_SPEC).
TIER_BOUNDS = {
    "easy": common.loose_bounds_deg,
    "medium": common.strict_bounds_deg,
    "hard": common.strict_bounds_deg,
}


def _cache_path(tier, spec=common.ARM_SPEC):
    suffix = "" if spec.key == "smplx_arm" else f"_{spec.key}"
    return os.path.join(_PKG_DIR, f"compare_targets{suffix}_{tier}.json")


def _sample_angles(bounds_deg, rng):
    flat = common.flat_bounds_deg(bounds_deg)
    lo = np.radians(np.asarray([b[0] for b in flat], dtype=np.float64))
    hi = np.radians(np.asarray([b[1] for b in flat], dtype=np.float64))
    return rng.uniform(lo, hi).astype(np.float32)


def _fk_target(fk_solver, angles, spec):
    fk = fk_solver.compute_fk_from_angles(angles)
    head, tail = fk_solver.get_bone_head_tail_from_fk(fk, spec.ee_bone)
    idx = fk_solver.bone_names.index(spec.ee_bone)
    R = np.asarray(fk[idx][:3, :3], dtype=np.float32)
    p = head if spec.ee_use_head else tail
    return np.asarray(p, dtype=np.float32), R


def _verify_reachable(jax_solver, tier, pos, R, spec):
    from jax_ik.objectives import DistanceObjTraj, EndEffectorOrientationObj

    mandatory = [
        DistanceObjTraj(
            bone_name=spec.ee_bone, target_points=pos, use_head=spec.ee_use_head, weight=1.0
        )
    ]
    if tier == "hard":
        mandatory.append(
            EndEffectorOrientationObj(bone_name=spec.ee_bone, target_transform=_r_to_4x4(R), weight=1.0)
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


def _r_to_4x4(R):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T


def generate_or_load(tier, n=NUM_TARGETS, verbose=True, spec=common.ARM_SPEC):
    path = _cache_path(tier, spec)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("n") == n and len(data.get("targets", [])) == n:
            return data["targets"]

    bounds_deg = tier_bounds_deg(tier, spec)
    fk_solver = common.FKSolver(
        model_file=spec.model_file, controlled_bones=spec.controlled_bones, do_compute_sdf=False
    )
    jax_solver = common.build_jax_ik_solver(
        bounds_deg,
        threshold=TIER_THRESHOLD[tier],
        model_file=spec.model_file,
        controlled_bones=spec.controlled_bones,
    )

    tier_offset = {"easy": 0, "medium": 1, "hard": 2}[tier]
    rng = np.random.default_rng(SEED_BASE + tier_offset)
    targets = []
    attempts = 0
    while len(targets) < n:
        attempts += 1
        angles = _sample_angles(bounds_deg, rng)
        pos, R = _fk_target(fk_solver, angles, spec)
        if _verify_reachable(jax_solver, tier, pos, R, spec):
            entry = {"pos": pos.tolist()}
            if tier == "hard":
                entry["R"] = R.tolist()
            targets.append(entry)
            if verbose and len(targets) % 20 == 0:
                print(f"[{spec.key}/{tier}] {len(targets)}/{n} verified (attempts={attempts})")
        if attempts > n * 20:
            raise RuntimeError(f"[{spec.key}/{tier}] too many failed reachability attempts ({attempts})")

    with open(path, "w") as f:
        json.dump({"n": n, "seed": SEED_BASE, "targets": targets}, f)
    if verbose:
        print(f"[{spec.key}/{tier}] generated+verified {n} targets in {attempts} attempts -> {path}")
    return targets


if __name__ == "__main__":
    for tier in ["easy", "medium", "hard"]:
        generate_or_load(tier)
