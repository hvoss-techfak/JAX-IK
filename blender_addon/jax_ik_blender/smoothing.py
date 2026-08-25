"""Autosmooth: temporal smoothing for a baked sequence of per-frame joint
angles, to reduce frame-to-frame flicker (e.g. the solver settling into a
slightly different local optimum on adjacent frames, which is otherwise
invisible per-frame but reads as jitter once played back).

Deliberately has no `bpy` import: this is pure numpy so it can be unit
tested outside Blender, and so bridge.py's Blender-facing code stays free
of the smoothing math itself.
"""

import numpy as np

# amount=1.0 maps to this many frames of Gaussian sigma. Chosen so the
# full [0, 1] range covers "basically off" through "quite smooth" for
# typical animation frame rates, without exposing a second knob.
MAX_SIGMA_FRAMES = 4.0


def smooth_angle_sequence(angles: np.ndarray, amount: float) -> np.ndarray:
    """Smooth a (T, D) sequence of joint angles (radians, D per-frame
    values, T frames along axis 0) across frames.

    Args:
        angles: (T, D) array, one row per baked frame.
        amount: 0..1. 0 returns `angles` unchanged. 1 applies the
            strongest smoothing (widest Gaussian window). Values in
            between blend smoothly, since a wider window's weights fall
            off from the center rather than adding frames abruptly.

    Returns:
        (T, D) array, same shape and dtype as `angles`.
    """
    angles = np.asarray(angles)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0 or angles.shape[0] < 2:
        return angles

    sigma = amount * MAX_SIGMA_FRAMES
    radius = max(1, int(np.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    # Unwrap each channel along the time axis first: a solve landing at,
    # say, -179 deg one frame and +179 deg the next is the same physical
    # angle on opposite sides of the +-180 wrap, and naively averaging
    # them would produce a meaningless ~0 -- exactly the kind of solve-to-
    # solve jump autosmooth exists to smooth over, so this has to run
    # before the convolution, not after.
    unwrapped = np.unwrap(angles.astype(np.float64), axis=0)

    # Edge-clamped padding: the first/last frames are smoothed against
    # repeats of themselves, not pulled toward zero or wrapped around,
    # which a bake's start/end frames should not be affected by.
    padded = np.pad(unwrapped, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.zeros_like(unwrapped)
    for i, w in enumerate(kernel):
        smoothed += w * padded[i : i + unwrapped.shape[0]]

    return smoothed.astype(angles.dtype)
