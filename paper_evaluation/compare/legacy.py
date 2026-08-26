"""Runs the four (three, after dropping IPOPT -- see run_compare.py) legacy
standalone IK scripts that already exist in paper_evaluation/ and produced
the original table_all.png: ablation.py --solver_type {ccd,fabrik} and
ik_tensorflow.py. Each is its own self-contained old FK/solver
implementation that predates the current src/jax_ik rewrite -- invoked
as-is via subprocess, one call per (algorithm, tier) handling all 100
targets internally (plus 11 warmup repeats of target 0, matching these
scripts' own "first 10 iterations are warmup" convention: they only print
"Time for iteration i" for i>10).

Output parsing replicates timing.py's run_script_full regex. One
discrepancy from timing.py's own math, by design: timing.py computes
"time per iteration" as (per-target wall time / target's list index i),
which shrinks as i grows and isn't a per-solver-step time at all -- reading
ablation.py/ik_tensorflow.py's source directly shows the printed
"Time for iteration i: ... seconds" is already that single target's own
solve wall-time (time_iter is reset every loop iteration, not accumulated),
so here time_per_iter_ms = solving_time_ms / steps, matching the
consistent definition used for jax_ik/ikpy/roboticstoolbox in this new
comparison.

"Success" for these four scripts is taken directly from their own printed
Success: True/False (their internal threshold check) -- NOT re-verified
through jax_ik's FK the way ikpy/roboticstoolbox/jax_ik itself are, since
these scripts reproduce the original paper's methodology verbatim and use
a different (older) FK implementation entirely.
"""

import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG_DIR)
MODEL_FILE = os.path.join(_ROOT, "smplx.glb")

LINE_RE = re.compile(
    r"Time for iteration\s+(\d+):\s+([\d.]+)\s+seconds\. Steps:\s+(\d+)\. Success:\s+(True|False)"
)

WARMUP = 11  # scripts only print for i>10, i.e. skip the first 11 (0..10)

CONTROLLED_BONES = "left_collar,left_shoulder,left_elbow,left_wrist"
DEFAULT_CONTROLLED_BONES = ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]


def _bounds_json(bounds_per_bone_deg):
    flat = []
    for lo, hi in bounds_per_bone_deg:
        for l, h in zip(lo, hi):
            flat.append([l, h])
    return json.dumps(flat)


def _run(script, extra_args, targets_pos, timeout=1800.0, model_file=MODEL_FILE, controlled_bones=DEFAULT_CONTROLLED_BONES):
    padded = [targets_pos[0]] * WARMUP + targets_pos
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = (
        [sys.executable, script]
        + ["--gltf_file", model_file, "--hand", "left"]
        + ["--controlled_bones", ",".join(controlled_bones)]
        + extra_args
        + ["--target_points", json.dumps(padded)]
    )
    result = subprocess.run(
        cmd,
        cwd=_PKG_DIR,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    rows = []
    for line in output.splitlines():
        m = LINE_RE.search(line)
        if m:
            i = int(m.group(1))
            time_s = float(m.group(2))
            steps = int(m.group(3))
            success = m.group(4) == "True"
            rows.append((i, time_s, steps, success))

    if result.returncode != 0 and not rows:
        raise RuntimeError(
            f"{script} {extra_args} failed (rc={result.returncode}):\n{output[-4000:]}"
        )

    rows.sort(key=lambda r: r[0])
    return rows, output


def run_ccd_or_fabrik(
    solver_type, tier, bounds_per_bone_deg, targets_pos, custom_objective,
    model_file=MODEL_FILE, controlled_bones=DEFAULT_CONTROLLED_BONES,
):
    extra = [
        "--bounds", _bounds_json(bounds_per_bone_deg),
        "--threshold", "0.005",
        "--num_steps", "500",
        "--max_iterations", "500",
        "--learning_rate", "0.1",
        "--solver_type", solver_type,
    ]
    if custom_objective:
        extra += ["--custom_objective", "True"]
    rows, output = _run(
        os.path.join(_PKG_DIR, "ablation.py"), extra, targets_pos,
        model_file=model_file, controlled_bones=controlled_bones,
    )
    return rows


def run_tensorflow(
    tier, bounds_per_bone_deg, targets_pos, additional_objective_weight,
    model_file=MODEL_FILE, controlled_bones=DEFAULT_CONTROLLED_BONES,
):
    extra = [
        "--bounds", _bounds_json(bounds_per_bone_deg),
        "--threshold", "0.005",
        "--num_steps", "500",
        "--max_iterations", "500",
        "--learning_rate", "0.1",
        "--additional_objective_weight", str(additional_objective_weight),
        "--subpoints", "0",
        "--cpu_only",
    ]
    rows, output = _run(
        os.path.join(_PKG_DIR, "ik_tensorflow.py"), extra, targets_pos,
        model_file=model_file, controlled_bones=controlled_bones,
    )
    return rows


def run_ipopt(tier, bounds_per_bone_deg, targets_pos, additional_objective_weight):
    extra = [
        "--bounds", _bounds_json(bounds_per_bone_deg),
        "--threshold", "0.005",
        "--num_steps", "500",
        "--max_iterations", "500",
        "--learning_rate", "0.1",
        "--additional_objective_weight", str(additional_objective_weight),
        "--cpu_only",
    ]
    rows, output = _run(os.path.join(_PKG_DIR, "ik_ipopt.py"), extra, targets_pos)
    return rows
