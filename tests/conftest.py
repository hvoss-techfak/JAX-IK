import os
import sys
import numpy as np
import pytest
import urllib.request

# Ensure local src/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from jax_ik.smplx_statics import left_arm_bounds_dict  # added

SMPLX_FILENAME = "smplx.glb"
SMPLX_URL = "https://uni-bielefeld.sciebo.de/s/B5StwQdiR4DW5mc/download"

def _ensure_smplx_model(path: str):
    """Ensure smplx.glb exists; attempt download if missing.
    Skips tests gracefully if download fails.
    Set env JAX_IK_SKIP_DOWNLOAD=1 to disable auto download.
    """
    if os.path.exists(path):
        return
    if os.environ.get("JAX_IK_SKIP_DOWNLOAD", "0") == "1":
        pytest.skip(f"{SMPLX_FILENAME} missing and auto-download disabled")
    try:
        print(f"Attempting download of {SMPLX_FILENAME} (~may take a moment)...")
        with urllib.request.urlopen(SMPLX_URL, timeout=60) as resp:  # nosec B310
            data = resp.read()
        if not data:
            raise RuntimeError("empty response")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Downloaded {SMPLX_FILENAME} ({len(data)} bytes)")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{SMPLX_FILENAME} not available and download failed: {e}")

# Central model path fixture
@pytest.fixture(scope="session")
def model_path():
    path = os.path.abspath(os.path.join(PROJECT_ROOT, SMPLX_FILENAME))
    _ensure_smplx_model(path)
    if not os.path.exists(path):
        pytest.skip(f"{SMPLX_FILENAME} not available – skipping IK tests that need real model")
    return path

@pytest.fixture(scope="session")
def basic_controlled_bones():
    return ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]

@pytest.fixture(scope="session")
def angle_dim(basic_controlled_bones):
    return len(basic_controlled_bones) * 3


def _build_bounds(controlled_bones):
    bounds = []
    for bone in controlled_bones:
        if bone not in left_arm_bounds_dict:
            raise KeyError(f"Bone {bone} not found in left_arm_bounds_dict")
        lower, upper = left_arm_bounds_dict[bone]
        for l,u in zip(lower, upper):
            bounds.append((l, u))  # degrees; solver converts to radians
    return bounds

@pytest.fixture(scope="session")
def solver(model_path, basic_controlled_bones):
    try:
        from jax_ik.ik import InverseKinematicsSolver
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Cannot import InverseKinematicsSolver (maybe optional dep missing like pyvista): {e}")
    bounds = _build_bounds(basic_controlled_bones)
    solver = InverseKinematicsSolver(
        model_file=model_path,
        controlled_bones=basic_controlled_bones,
        bounds=bounds,
        threshold=1e-6,
        num_steps=60,
        compute_sdf=False,
    )
    return solver

@pytest.fixture()
def zero_angles(angle_dim):
    return np.zeros(angle_dim, dtype=np.float32)

@pytest.fixture()
def two_frame_traj(zero_angles):
    return np.stack([zero_angles, zero_angles.copy()], axis=0)
