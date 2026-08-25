"""Unit tests for smoothing.smooth_angle_sequence. Pure numpy, no bpy --
run directly with `python smoothing_test.py` (unlike the other tests in
this directory, which need a real Blender + the add-on installed).
"""

import importlib.util
import os
import sys

import numpy as np

# Loaded by file path, not `from jax_ik_blender.smoothing import ...`:
# importing the jax_ik_blender *package* runs its __init__.py, which pulls
# in bridge.py and friends -- all of which import bpy. smoothing.py itself
# has no bpy dependency, so this lets it be tested without a Blender
# install, matching this test's whole point.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smoothing", os.path.join(_here, "..", "jax_ik_blender", "smoothing.py")
)
_smoothing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoothing)
smooth_angle_sequence = _smoothing.smooth_angle_sequence

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# --- amount=0 is a no-op -----------------------------------------------
rng = np.random.default_rng(0)
raw = rng.uniform(-1.0, 1.0, size=(20, 6)).astype(np.float32)
out = smooth_angle_sequence(raw, 0.0)
check("amount=0 leaves the sequence unchanged", np.array_equal(out, raw))

# --- a single noisy spike gets pulled toward its neighbors --------------
T, D = 30, 3
base = np.zeros((T, D), dtype=np.float32)
base[15, 0] = 1.0  # one frame jumps far off from its flat neighbors
smoothed = smooth_angle_sequence(base, 0.5)
check(
    "a lone spike is reduced by smoothing",
    smoothed[15, 0] < base[15, 0],
    f"spike {base[15, 0]} -> {smoothed[15, 0]:.4f}",
)
check(
    "smoothing does not touch an already-flat, unaffected channel",
    np.allclose(smoothed[:, 1], 0.0, atol=1e-6),
)
check(
    "a stronger amount smooths the spike more",
    smooth_angle_sequence(base, 1.0)[15, 0] < smooth_angle_sequence(base, 0.2)[15, 0],
)

# --- output shape/dtype match input -------------------------------------
check("output shape matches input", smoothed.shape == base.shape)
check("output dtype matches input", smoothed.dtype == base.dtype)

# --- edges are clamped, not pulled toward 0 or wrapped ------------------
const = np.full((10, 1), 2.5, dtype=np.float32)
smoothed_const = smooth_angle_sequence(const, 1.0)
check(
    "a constant sequence stays constant (edge clamping, not zero-padding)",
    np.allclose(smoothed_const, 2.5, atol=1e-4),
    f"{smoothed_const.ravel()}",
)

# --- angle wraparound: -179deg/+179deg (in radians) is *not* averaged to ~0
deg = np.pi / 180.0
wrap_seq = np.array(
    [[178 * deg], [179 * deg], [-179 * deg], [-178 * deg], [-177 * deg]], dtype=np.float32
)
smoothed_wrap = smooth_angle_sequence(wrap_seq, 0.3)
check(
    "wraparound angles are unwrapped before smoothing, not naively averaged",
    np.all(np.abs(np.abs(smoothed_wrap) - np.pi) < 0.2),
    f"{smoothed_wrap.ravel()}",
)

# --- too-short sequences are returned unchanged rather than erroring ----
one_frame = np.array([[0.1, 0.2]], dtype=np.float32)
check(
    "a single-frame sequence is returned unchanged",
    np.array_equal(smooth_angle_sequence(one_frame, 1.0), one_frame),
)

# --- amount is clamped to [0, 1] ----------------------------------------
try:
    smooth_angle_sequence(base, 5.0)
    smooth_angle_sequence(base, -1.0)
    check("out-of-range amount does not raise", True)
except Exception as exc:  # noqa: BLE001
    check("out-of-range amount does not raise", False, str(exc))

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
