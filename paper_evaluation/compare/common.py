"""Shared chain model + bounds for the jax-ik vs. external-package comparison.

Builds kinematic chains for ikpy and roboticstoolbox that are mathematically
identical to jax_ik's own FK, for any SMPL-X ball-joint bone chain (each
controlled bone = 3 sequential revolute joints X,Y,Z, matching jax_ik's
per-bone Euler convention) -- so that any target FK-sampled as reachable in
jax_ik is reachable in the other chains too. Originally written for the
4-bone left-arm chain (evaluation 1, frozen: compare_results.csv /
compare_table.png); generalized here to take controlled_bones/model_file/
ee_bone as parameters so evaluation 2 (7-bone finger chain) can reuse it
without duplicating logic. Every function keeps its evaluation-1 default
arguments so run_compare.py (already executed) keeps working unmodified.

UR5 (evaluation 3) is NOT built with this module: it's a true single-DOF-
per-joint URDF robot, not a 3-DOF-ball-joint chain, so it uses native ikpy/
roboticstoolbox URDF loaders instead -- see ur5_common.py.

jax_ik's FK (src/jax_ik/ik.py):
    G_bone = G_parent @ Local_bone @ R_bone(angles)
    R_bone(angles) = Rz(angles[2]) @ Ry(angles[1]) @ Rx(angles[0])   (see
    tf_euler_to_matrix -- closed form for exactly this product)
    tail = G_bone @ [0, bone_length, 0, 1]
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
from jax_ik.smplx_statics import left_arm_bounds_dict  # noqa: E402

MODEL_FILE = os.path.join(_ROOT, "smplx.glb")
CONTROLLED_BONES = ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]
EE_BONE = "left_wrist"

# 7-bone finger chain (evaluation 2): left_arm_bounds_dict already has all
# of these entries (it was written to cover the whole arm+hand).
FINGER_CONTROLLED_BONES = [
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "left_index1", "left_index2", "left_index3",
]
FINGER_EE_BONE = "left_index3"

LOOSE_DEG = 179.0


class ModelSpec:
    """Bundles what a "smplx"-kind (3-DOF-ball-joint chain) evaluation needs.
    UR5 doesn't use this -- see ur5_common.py's own constants."""

    def __init__(self, key, model_file, controlled_bones, ee_bone, ee_use_head=False):
        self.key = key
        self.model_file = model_file
        self.controlled_bones = controlled_bones
        self.ee_bone = ee_bone
        self.ee_use_head = ee_use_head


ARM_SPEC = ModelSpec("smplx_arm", MODEL_FILE, CONTROLLED_BONES, EE_BONE)
FINGER_SPEC = ModelSpec("smplx_fingers", MODEL_FILE, FINGER_CONTROLLED_BONES, FINGER_EE_BONE)


def strict_bounds_deg(controlled_bones=CONTROLLED_BONES):
    """[(lo3, hi3), ...] per bone, degrees -- the real anatomical limits."""
    out = []
    for bone in controlled_bones:
        lo, hi = left_arm_bounds_dict[bone]
        out.append((list(lo), list(hi)))
    return out


def loose_bounds_deg(controlled_bones=CONTROLLED_BONES):
    """[(lo3, hi3), ...] per bone, degrees -- effectively unconstrained."""
    return [([-LOOSE_DEG] * 3, [LOOSE_DEG] * 3) for _ in controlled_bones]


def flat_bounds_deg(bounds_per_bone):
    """Flatten to the (lo,hi) tuple-per-axis list InverseKinematicsSolver wants."""
    flat = []
    for lo, hi in bounds_per_bone:
        for l, h in zip(lo, hi):
            flat.append((l, h))
    return flat


def build_jax_ik_solver(
    bounds_per_bone_deg,
    threshold=0.005,
    num_steps=500,
    compute_sdf=False,
    model_file=MODEL_FILE,
    controlled_bones=CONTROLLED_BONES,
):
    return InverseKinematicsSolver(
        model_file=model_file,
        controlled_bones=controlled_bones,
        bounds=flat_bounds_deg(bounds_per_bone_deg),
        threshold=threshold,
        num_steps=num_steps,
        compute_sdf=compute_sdf,
    )


def bone_offsets_and_lengths(fk_solver, controlled_bones=CONTROLLED_BONES):
    """Per-bone translation offset (3,) and bone_length, in controlled_bones order.

    controlled_bones[0] is not a root: the skeleton keeps its ancestors
    (pelvis/spine/...) too, so its *own* local_transform only covers the
    offset from its immediate parent, not the full offset from the world
    origin. Since every bone here has identity rest rotation (asserted
    below) and at all-zero controlled angles every controlled bone's own
    rotation is also identity, FK at the zero pose gives exactly
    G_ancestors @ Local_bone0 for free -- use that as bone 0's offset so
    the replicated chain starts from the same world position jax_ik does.
    Bones 1..N-1 are each the direct parent of the next controlled bone
    (true for both the 4-bone arm and the 7-bone arm+finger chain), so
    their own local_transform translation is already the right inter-bone
    offset.
    """
    offsets, lengths = [], []
    for i, bone in enumerate(controlled_bones):
        b = fk_solver.skeleton[bone]
        local = np.asarray(b["local_transform"], dtype=np.float64)
        # Sanity: rotation part must be identity (see module docstring) --
        # if this ever fires, the chain-folding trick above no longer holds
        # and origin_orientation would need a real rpy decomposition.
        assert np.allclose(local[:3, :3], np.eye(3), atol=1e-5), (
            f"{bone} local_transform has non-identity rotation; "
            "chain construction assumption violated"
        )
        if i == 0:
            zero = np.zeros(len(controlled_bones) * 3, dtype=np.float32)
            fk0 = fk_solver.compute_fk_from_angles(zero)
            idx = fk_solver.bone_names.index(bone)
            G0 = np.asarray(fk0[idx])
            assert np.allclose(G0[:3, :3], np.eye(3), atol=1e-5), (
                f"ancestor chain of {bone} has non-identity rest rotation; "
                "chain construction assumption violated"
            )
            offsets.append(G0[:3, 3].astype(np.float64))
        else:
            offsets.append(local[:3, 3])
        lengths.append(b["bone_length"])
    return offsets, lengths


# --------------------------------------------------------------------------
# ikpy chain
# --------------------------------------------------------------------------
def build_ikpy_chain(bounds_per_bone_deg, offsets, lengths, controlled_bones=CONTROLLED_BONES):
    import ikpy.chain
    import ikpy.link

    zero = np.zeros(3)
    links = [ikpy.link.OriginLink()]
    active_mask = [False]
    for i, bone in enumerate(controlled_bones):
        lo_deg, hi_deg = bounds_per_bone_deg[i]
        lo = np.radians(lo_deg)
        hi = np.radians(hi_deg)
        off = offsets[i]
        # Order Z, Y, X -> composes (intrinsically) as Rz @ Ry @ Rx, matching
        # tf_euler_to_matrix exactly.
        links.append(
            ikpy.link.URDFLink(
                name=f"{bone}_z",
                origin_translation=off,
                origin_orientation=zero,
                rotation=[0, 0, 1],
                bounds=(lo[2], hi[2]),
            )
        )
        links.append(
            ikpy.link.URDFLink(
                name=f"{bone}_y",
                origin_translation=zero,
                origin_orientation=zero,
                rotation=[0, 1, 0],
                bounds=(lo[1], hi[1]),
            )
        )
        links.append(
            ikpy.link.URDFLink(
                name=f"{bone}_x",
                origin_translation=zero,
                origin_orientation=zero,
                rotation=[1, 0, 0],
                bounds=(lo[0], hi[0]),
            )
        )
        active_mask += [True, True, True]

    links.append(
        ikpy.link.URDFLink(
            name="ee_tail",
            origin_translation=[0.0, lengths[-1], 0.0],
            origin_orientation=zero,
            rotation=None,
            translation=None,
            joint_type="fixed",
        )
    )
    active_mask.append(False)

    chain = ikpy.chain.Chain(links, active_links_mask=active_mask)
    return chain


# --------------------------------------------------------------------------
# roboticstoolbox chain (ETS)
# --------------------------------------------------------------------------
def build_rtb_robot(bounds_per_bone_deg, offsets, lengths, controlled_bones=CONTROLLED_BONES):
    import roboticstoolbox as rtb

    # Explicit joint indices (jindex=...): building each bone's segment as
    # an independent ETS and multiplying them together restarts jindex
    # numbering at 0 for each segment, which Robot() rejects as a repeated
    # index -- so number every revolute joint globally ourselves.
    E = None
    jidx = 0
    for i, bone in enumerate(controlled_bones):
        lo_deg, hi_deg = bounds_per_bone_deg[i]
        lo = np.radians(lo_deg)
        hi = np.radians(hi_deg)
        ox, oy, oz = offsets[i]
        seg = rtb.ET.tx(float(ox)) * rtb.ET.ty(float(oy)) * rtb.ET.tz(float(oz))
        seg = seg * rtb.ET.Rz(qlim=[float(lo[2]), float(hi[2])], jindex=jidx)
        jidx += 1
        seg = seg * rtb.ET.Ry(qlim=[float(lo[1]), float(hi[1])], jindex=jidx)
        jidx += 1
        seg = seg * rtb.ET.Rx(qlim=[float(lo[0]), float(hi[0])], jindex=jidx)
        jidx += 1
        E = seg if E is None else E * seg
    E = E * rtb.ET.ty(float(lengths[-1]))
    robot = rtb.Robot(E)
    return robot


# --------------------------------------------------------------------------
# FK-agreement self-check
# --------------------------------------------------------------------------
def _jax_ik_fk(fk_solver, angles_rad_flat, ee_bone=EE_BONE, use_head=False):
    fk = fk_solver.compute_fk_from_angles(np.asarray(angles_rad_flat, dtype=np.float32))
    head, tail = fk_solver.get_bone_head_tail_from_fk(fk, ee_bone)
    idx = fk_solver.bone_names.index(ee_bone)
    R = np.asarray(fk[idx][:3, :3])
    p = head if use_head else tail
    return np.asarray(p), R


def xyz_to_zyx(angles_flat):
    """jax_ik orders each bone's 3 angles [x,y,z]; the external chains below
    add joints in order Z,Y,X per bone (so their composed rotation is
    Rz@Ry@Rx, matching tf_euler_to_matrix) -- so angle vectors handed to
    their forward_kinematics/fkine/ikine calls must be reordered to
    [z,y,x] per bone first. Inverse of zyx_to_xyz."""
    a = np.asarray(angles_flat).reshape(-1, 3)
    return a[:, [2, 1, 0]].reshape(-1)


def zyx_to_xyz(angles_flat):
    a = np.asarray(angles_flat).reshape(-1, 3)
    return a[:, [2, 1, 0]].reshape(-1)


def _ikpy_fk(chain, angles_rad_flat):
    full = np.concatenate([[0.0], xyz_to_zyx(angles_rad_flat), [0.0]])
    T = chain.forward_kinematics(full)
    return T[:3, 3].copy(), T[:3, :3].copy()


def _rtb_fk(robot, angles_rad_flat):
    T = robot.fkine(xyz_to_zyx(angles_rad_flat)).A
    return T[:3, 3].copy(), T[:3, :3].copy()


def check_fk_agreement(
    bounds_per_bone_deg,
    n=20,
    seed=0,
    atol=1e-4,
    model_file=MODEL_FILE,
    controlled_bones=CONTROLLED_BONES,
    ee_bone=EE_BONE,
):
    """Sample n random angle vectors within bounds and assert ikpy/rtb FK
    agree with jax_ik's own FK (position and orientation) to within atol.
    Returns (ikpy_chain, rtb_robot, fk_solver) on success; raises otherwise.
    """
    fk_solver = FKSolver(
        model_file=model_file, controlled_bones=controlled_bones, do_compute_sdf=False
    )
    offsets, lengths = bone_offsets_and_lengths(fk_solver, controlled_bones)

    ikpy_chain = build_ikpy_chain(bounds_per_bone_deg, offsets, lengths, controlled_bones)
    rtb_robot = build_rtb_robot(bounds_per_bone_deg, offsets, lengths, controlled_bones)

    lo = np.radians(np.asarray([b[0] for b in bounds_per_bone_deg]).reshape(-1))
    hi = np.radians(np.asarray([b[1] for b in bounds_per_bone_deg]).reshape(-1))

    rng = np.random.default_rng(seed)
    for _ in range(n):
        angles = rng.uniform(lo, hi)
        p_ref, R_ref = _jax_ik_fk(fk_solver, angles, ee_bone=ee_bone)
        p_ikpy, R_ikpy = _ikpy_fk(ikpy_chain, angles)
        p_rtb, R_rtb = _rtb_fk(rtb_robot, angles)

        assert np.allclose(p_ref, p_ikpy, atol=atol), (p_ref, p_ikpy)
        assert np.allclose(R_ref, R_ikpy, atol=atol), (R_ref, R_ikpy)
        assert np.allclose(p_ref, p_rtb, atol=atol), (p_ref, p_rtb)
        assert np.allclose(R_ref, R_rtb, atol=atol), (R_ref, R_rtb)

    return ikpy_chain, rtb_robot, fk_solver
