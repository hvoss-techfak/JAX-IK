"""UR5 (models/UR5.urdf) model definitions for evaluation 3.

UR5 is a true single-DOF-per-joint URDF robot, unlike SMPL-X's 3-DOF ball
joints, so it does NOT reuse common.py's manual ball-joint chain builder --
it uses ikpy's and roboticstoolbox's own native URDF loaders instead (per
the coordinator's explicit instruction), reconciling jax_ik's own URDF
coordinate-frame quirk (see below) rather than hand-building link geometry.

--- Coordinate-frame reconciliation ---
jax_ik's URDF loader (src/jax_ik/helper.py load_skeleton_from_urdf) bakes a
FIXED transform onto the skeleton's root link only:
    coord_transform = translate(0,-1,0) @ rotate(+90, Y) @ rotate(-90, X)
Since every other link's local_transform is untouched (just the raw URDF
joint origins), and FK composes G_child = G_parent @ Local_child, this
collapses to a single global rigid transform on the WHOLE tree:
    G_link_jax = coord_transform @ G_link_native
Verified numerically against jax_ik's own FK for tool0 at the zero pose
(see git history / development notes) -- p_jax = R @ p_native + t,
R_jax = R @ R_native, with R = coord_transform[:3,:3], t = coord_transform[:3,3].

ikpy/roboticstoolbox load the URDF natively (their own loaders don't know
about jax_ik's transform), so they FK/IK in the *native* URDF frame. Rather
than rebuild their chains with a prefixed transform, targets generated in
jax_ik's frame are converted to the native frame (inverse transform) before
being handed to ikpy/roboticstoolbox; the resulting joint angles are then
handed back to jax_ik's own FK (which re-applies coord_transform correctly)
for the uniform success re-check, so no inverse conversion is needed there.

--- DOF reconciliation ---
jax_ik still represents each controlled bone with a full 3-DOF Euler angle
triplet (see ik.py's bounds auto-derivation), even though only one axis per
UR5 joint is actually free (the other two are pinned to jax_ik's own
default +-10 degree slack, see ik.py ~1095-1152). ikpy/roboticstoolbox's
native URDF chains are strictly single-DOF per joint and cannot represent
that secondary-axis slack at all. To keep target generation exactly
reachable by both representations, targets are FK-sampled with the
secondary axes forced to exactly 0 (a pure single-DOF pose), which an
ikpy/roboticstoolbox single-axis chain reproduces bit-for-bit; jax_ik's own
*solver* is still left free to use its full 18-DOF (including the +-10
degree slack) when actually solving, same as it would on a real deployment
-- "success" is always re-verified through jax_ik's own FK/threshold
regardless of which representation produced the angles.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG_DIR)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from jax_ik.ik import FKSolver, InverseKinematicsSolver  # noqa: E402
from . import common  # noqa: E402

UR5_PATH = os.path.join(_ROOT, "models", "UR5.urdf")
UR5_NOMESH_PATH = os.path.join(_HERE, "_ur5_nomesh.urdf")

UR5_BONES = [
    "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
]
EE_LINK = "tool0"

# For reusing algos.solve_jax_ik/reverify_success unchanged: those only
# read spec.ee_bone/spec.ee_use_head, never spec.model_file/controlled_bones
# (the solver/fk_solver are built separately here via build_jax_ik_solver),
# so this is safe to share even though UR5 doesn't use the rest of common.py.
UR5_SPEC = common.ModelSpec("ur5", UR5_PATH, UR5_BONES, EE_LINK, ee_use_head=True)

# axis index (0=X,1=Y,2=Z) of each joint's real rotation axis, confirmed
# from UR5.urdf's own <axis xyz=.../> elements and cross-checked against
# jax_ik's own bounds auto-derivation output.
PRIMARY_AXIS = {
    "shoulder_link": 2,  # shoulder_pan_joint, axis Z
    "upper_arm_link": 1,  # shoulder_lift_joint, axis Y
    "forearm_link": 1,  # elbow_joint, axis Y
    "wrist_1_link": 1,  # wrist_1_joint, axis Y
    "wrist_2_link": 2,  # wrist_2_joint, axis Z
    "wrist_3_link": 1,  # wrist_3_joint, axis Y
}
SECONDARY_DEG = 10.0  # jax_ik's own default slack on the two non-primary axes

# ±90 degrees: a common conservative "collaborative/industrial" joint range
# (well inside the UR5's real ±360 mechanical limits, and a genuine
# contrast with the ±180 "easy" bound -- jax_ik's own URDF auto-derivation
# already clamps to ±180, so ±180 doubles as the natural "loose" tier).
STRICT_DEG = 90.0

_ROT_X = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
_ROT_Y = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
_TRANSLATE = np.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
COORD_TRANSFORM = _TRANSLATE @ _ROT_Y @ _ROT_X
COORD_R = COORD_TRANSFORM[:3, :3]
COORD_T = COORD_TRANSFORM[:3, 3]


def native_from_jax(pos_jax, R_jax=None):
    pos_native = COORD_R.T @ (np.asarray(pos_jax, dtype=np.float64) - COORD_T)
    if R_jax is None:
        return pos_native, None
    R_native = COORD_R.T @ np.asarray(R_jax, dtype=np.float64)
    return pos_native, R_native


def _ensure_nomesh_urdf():
    """ikpy/roboticstoolbox URDF loaders choke on this file's
    package://ur_description/... mesh URIs (no ROS package registered) --
    we only need kinematics, so cache a copy with <visual>/<collision>
    stripped."""
    if os.path.exists(UR5_NOMESH_PATH):
        return UR5_NOMESH_PATH
    import xml.etree.ElementTree as ET

    tree = ET.parse(UR5_PATH)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                link.remove(el)
    tree.write(UR5_NOMESH_PATH)
    return UR5_NOMESH_PATH


def full18_bounds_deg(primary_bounds_deg_6):
    """Expand 6 per-joint (lo,hi) primary-axis bounds into jax_ik's own
    18-entry [x,y,z]-per-bone convention (secondary axes = +-SECONDARY_DEG)."""
    out = []
    for bone, (plo, phi) in zip(UR5_BONES, primary_bounds_deg_6):
        axis3 = [(-SECONDARY_DEG, SECONDARY_DEG)] * 3
        axis3[PRIMARY_AXIS[bone]] = (plo, phi)
        out.extend(axis3)
    return out


def build_jax_ik_solver(primary_bounds_deg_6, threshold=0.005, num_steps=500):
    bounds = None if primary_bounds_deg_6 is None else full18_bounds_deg(primary_bounds_deg_6)
    return InverseKinematicsSolver(
        model_file=UR5_PATH,
        controlled_bones=UR5_BONES,
        bounds=bounds,
        threshold=threshold,
        num_steps=num_steps,
        compute_sdf=False,
    )


def angle18_from_q6(q6):
    """Expand a 6-vector (one angle per joint, native single-DOF) into
    jax_ik's 18-entry convention with secondary axes exactly 0."""
    q6 = np.asarray(q6, dtype=np.float32)
    out = np.zeros(18, dtype=np.float32)
    for i, bone in enumerate(UR5_BONES):
        out[3 * i + PRIMARY_AXIS[bone]] = q6[i]
    return out


def q6_from_angle18(angles18):
    """Extract the primary-axis component per bone from jax_ik's 18-entry
    angle vector (used to feed a jax_ik solve result back into an
    ikpy/roboticstoolbox single-DOF chain, e.g. for warm starts -- not used
    for success re-verification, which always goes through jax_ik's own FK
    on the full 18-dim vector instead)."""
    angles18 = np.asarray(angles18, dtype=np.float32).reshape(6, 3)
    return np.asarray(
        [angles18[i, PRIMARY_AXIS[bone]] for i, bone in enumerate(UR5_BONES)], dtype=np.float32
    )


# --------------------------------------------------------------------------
# ikpy (native URDF loader)
# --------------------------------------------------------------------------
_IKPY_BASE_ELEMENTS = [
    "base_link", "shoulder_pan_joint", "shoulder_link",
    "shoulder_lift_joint", "upper_arm_link",
    "elbow_joint", "forearm_link",
    "wrist_1_joint", "wrist_1_link",
    "wrist_2_joint", "wrist_2_link",
    "wrist_3_joint", "wrist_3_link",
    "wrist_3_link-tool0_fixed_joint", "tool0",
]


def build_ikpy_chain(primary_bounds_deg_6):
    import ikpy.chain

    chain = ikpy.chain.Chain.from_urdf_file(
        UR5_PATH,
        base_elements=_IKPY_BASE_ELEMENTS,
        active_links_mask=[False, True, True, True, True, True, True, False],
    )
    for i in range(6):
        lo, hi = primary_bounds_deg_6[i]
        chain.links[i + 1].bounds = (np.radians(lo), np.radians(hi))
    return chain


def _ikpy_fk(chain, q6):
    full = np.concatenate([[0.0], np.asarray(q6, dtype=np.float64), [0.0]])
    T = chain.forward_kinematics(full)
    return T[:3, 3].copy(), T[:3, :3].copy()


# --------------------------------------------------------------------------
# roboticstoolbox (native URDF loader, mesh-stripped copy)
# --------------------------------------------------------------------------
def build_rtb_robot(primary_bounds_deg_6):
    import roboticstoolbox as rtb

    path = _ensure_nomesh_urdf()
    robot = rtb.Robot.URDF(path)
    lo = np.radians(np.asarray([b[0] for b in primary_bounds_deg_6]))
    hi = np.radians(np.asarray([b[1] for b in primary_bounds_deg_6]))
    robot.qlim = np.vstack([lo, hi])
    return robot


def _rtb_fk(robot, q6):
    T = robot.fkine(np.asarray(q6, dtype=np.float64), end=EE_LINK).A
    return T[:3, 3].copy(), T[:3, :3].copy()


# --------------------------------------------------------------------------
# FK-agreement self-check
# --------------------------------------------------------------------------
def _jax_ik_fk(fk_solver, q6):
    angles18 = angle18_from_q6(q6)
    fk = fk_solver.compute_fk_from_angles(angles18)
    idx = fk_solver.bone_names.index(EE_LINK)
    T = np.asarray(fk[idx])
    return T[:3, 3].copy(), T[:3, :3].copy()


def check_fk_agreement(primary_bounds_deg_6, n=20, seed=0, atol=1e-4):
    fk_solver = FKSolver(model_file=UR5_PATH, controlled_bones=UR5_BONES, do_compute_sdf=False)
    ikpy_chain = build_ikpy_chain(primary_bounds_deg_6)
    rtb_robot = build_rtb_robot(primary_bounds_deg_6)

    lo = np.radians(np.asarray([b[0] for b in primary_bounds_deg_6]))
    hi = np.radians(np.asarray([b[1] for b in primary_bounds_deg_6]))
    rng = np.random.default_rng(seed)

    for _ in range(n):
        q6 = rng.uniform(lo, hi)
        p_jax, R_jax = _jax_ik_fk(fk_solver, q6)
        p_native_expected, R_native_expected = native_from_jax(p_jax, R_jax)

        p_ikpy, R_ikpy = _ikpy_fk(ikpy_chain, q6)
        p_rtb, R_rtb = _rtb_fk(rtb_robot, q6)

        assert np.allclose(p_native_expected, p_ikpy, atol=atol), (p_native_expected, p_ikpy)
        assert np.allclose(R_native_expected, R_ikpy, atol=atol), (R_native_expected, R_ikpy)
        assert np.allclose(p_native_expected, p_rtb, atol=atol), (p_native_expected, p_rtb)
        assert np.allclose(R_native_expected, R_rtb, atol=atol), (R_native_expected, R_rtb)

    return ikpy_chain, rtb_robot, fk_solver
