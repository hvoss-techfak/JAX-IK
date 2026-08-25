"""Blender <-> jax_ik translation layer.

This module is the only place that talks to the `jax_ik` package directly.
Everything else in the add-on works in terms of Blender types (armatures,
pose bones, the JaxIKChain/JaxIKObjective property groups).

Coordinate space note: all FK math here happens in "armature-local" space,
i.e. `bone.matrix_local` / `pose_bone.matrix` as Blender already provides
them (no glTF-style Y-up conversion needed -- see the add-on's README for
why). World-space objects (target Empties) are converted into that space
with a single `armature.matrix_world.inverted() @ ...` multiply.
"""

import math
import os

import bpy
import mathutils
import numpy as np

_jax_ik_ik = None
_jax_ik_objectives = None

# Cache of constructed FKSolver instances, keyed by a cheap structural
# signature. This matters for correctness, not just speed: solve_ik's jitted
# core treats the FKSolver as a *static* jax.jit argument, and JAX caches
# compiled traces by static-argument identity/equality. FKSolver doesn't
# define __eq__/__hash__, so a freshly constructed instance never compares
# equal to a previous one -- if we built a new FKSolver on every solve, every
# single solve (including every tick of "Live" dragging) would force a full
# recompile of the optimizer loop. Reusing the same Python object for the
# same (armature, controlled_bones, bones_of_interest) signature is what
# `InverseKinematicsSolver._pruned_fk_cache` does internally in the library;
# we're not going through that class, so we replicate it here.
_fk_solver_cache = {}


def _ensure_jax_ik_imported():
    """Import jax_ik on first use, working around a footgun in jax_ik.ik:
    it does `os.makedirs("./jax_cache", ...)` and points JAX's compilation
    cache at that *relative* path at import time. Blender's cwd is not
    guaranteed to be writable (especially under Flatpak sandboxing), so we
    temporarily chdir into an add-on-owned, guaranteed-writable directory
    for the duration of the first import.
    """
    global _jax_ik_ik, _jax_ik_objectives
    if _jax_ik_ik is not None:
        return _jax_ik_ik, _jax_ik_objectives

    cache_dir = bpy.utils.user_resource(
        "SCRIPTS", path=os.path.join("jax_ik_blender", "jax_cache"), create=True
    )
    if not cache_dir:
        import tempfile

        cache_dir = os.path.join(tempfile.gettempdir(), "jax_ik_blender_jax_cache")
        os.makedirs(cache_dir, exist_ok=True)

    prev_cwd = os.getcwd()
    try:
        os.chdir(cache_dir)
        import jax_ik.ik as jax_ik_ik
        import jax_ik.objectives as jax_ik_objectives
    finally:
        os.chdir(prev_cwd)

    # jax_ik.ik points JAX's compilation cache at a *relative* "./jax_cache"
    # path (see module docstring above) -- that resolves against whatever
    # Blender's cwd happens to be at actual *compile* time (not import time,
    # which is why the chdir above isn't enough on its own), which can be a
    # read-only location. Re-point it at our absolute, writable cache_dir.
    import jax

    jax.config.update("jax_compilation_cache_dir", cache_dir)

    _jax_ik_ik = jax_ik_ik
    _jax_ik_objectives = jax_ik_objectives
    return _jax_ik_ik, _jax_ik_objectives


def is_available() -> tuple[bool, str]:
    """Try importing jax_ik and report whether it's available, without
    raising. Used by the diagnostics panel."""
    try:
        _ensure_jax_ik_imported()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - surfacing any import failure to the UI
        return False, str(exc)


_custom_objective_classes = {}


def _get_custom_objectives() -> dict:
    """Build (once) and return {"LiveAimObj": ..., "PoleTargetObj": ...} --
    two small ObjectiveFunction subclasses we define ourselves rather than
    reusing jax_ik's own BoneRelativeLookObj/BoneDirectionObjective.

    Why: those two classes compute their target direction/point *once*
    (either directly, for BoneDirectionObjective, or via a fixed offset from
    the bone's pre-solve head/tail, for BoneRelativeLookObj) and hold it
    fixed for the whole solve. That's fine if the bone doing the aiming is
    the only thing moving, but every bone *above* it in the same chain is
    also free during a solve -- and if any of them rotate, the bone's head
    moves too, which silently invalidates that frozen reference. In testing
    this reliably converged to a *low loss* pose that nonetheless pointed
    15+ degrees away from the actual target, exactly matching "Look At
    never aligns to the target" -- the loss and the true geometric error had
    quietly decoupled.

    LiveAimObj/PoleTargetObj fix this by never freezing anything that
    depends on the chain's own pose: they take the FK head/tail *inside*
    __call__ (so it's re-evaluated on every optimizer step) and compare it
    only against externally-fixed points (the target's world position,
    converted to armature space once -- which is fine, since that
    conversion doesn't depend on the chain's pose at all).
    """
    global _custom_objective_classes
    if _custom_objective_classes:
        return _custom_objective_classes

    _, jax_ik_objectives = _ensure_jax_ik_imported()
    import jax.numpy as jnp
    from jax.tree_util import register_pytree_node_class

    ObjectiveFunction = jax_ik_objectives.ObjectiveFunction

    def _safe_norm(x, eps=1e-6):
        return jnp.sqrt(jnp.sum(jnp.square(x)) + eps * eps)

    def _safe_arccos(x, eps=1e-6):
        return jnp.arccos(jnp.clip(x, -1.0 + eps, 1.0 - eps))

    @register_pytree_node_class
    class LiveAimObj(ObjectiveFunction):
        """Aim a bone's head->tail axis at a fixed world point, live."""

        LAST_FRAME_ONLY = True

        def __init__(self, bone_name, use_head, target_point, weight=1.0):
            self.bone_name = bone_name
            self.use_head = bool(use_head)
            self.target_point = jnp.asarray(target_point, jnp.float32)
            self.weight = jnp.asarray(weight, jnp.float32)

        def referenced_bones(self):
            return (self.bone_name,)

        def __call__(self, X, fk_solver):
            cfg = X if X.ndim == 1 else X[-1]
            fk = fk_solver.compute_fk_from_angles(cfg)
            head, tail = fk_solver.get_bone_head_tail_from_fk(fk, self.bone_name)
            ref = head if self.use_head else tail
            bone_vec = (tail - head) / _safe_norm(tail - head)
            tgt_vec = (self.target_point - ref) / _safe_norm(self.target_point - ref)
            cos_th = jnp.dot(bone_vec, tgt_vec)
            return _safe_arccos(cos_th) ** 2 * self.weight

    @register_pytree_node_class
    class PoleTargetObj(ObjectiveFunction):
        """Standard two-bone pole-vector constraint: the middle joint is
        pushed toward the half-plane (relative to the root->tip axis) that
        contains the pole target, controlling bend direction (elbow/knee).
        """

        LAST_FRAME_ONLY = True

        def __init__(self, root_bone, mid_bone, tip_bone, pole_point, weight=1.0):
            self.root_bone = root_bone
            self.mid_bone = mid_bone
            self.tip_bone = tip_bone
            self.pole_point = jnp.asarray(pole_point, jnp.float32)
            self.weight = jnp.asarray(weight, jnp.float32)

        def referenced_bones(self):
            return (self.root_bone, self.mid_bone, self.tip_bone)

        def __call__(self, X, fk_solver):
            cfg = X if X.ndim == 1 else X[-1]
            fk = fk_solver.compute_fk_from_angles(cfg)
            root_head, _ = fk_solver.get_bone_head_tail_from_fk(fk, self.root_bone)
            mid_head, _ = fk_solver.get_bone_head_tail_from_fk(fk, self.mid_bone)
            tip_head, _ = fk_solver.get_bone_head_tail_from_fk(fk, self.tip_bone)

            chain_axis = tip_head - root_head
            chain_axis = chain_axis / _safe_norm(chain_axis)

            mid_offset = mid_head - root_head
            mid_perp = mid_offset - jnp.dot(mid_offset, chain_axis) * chain_axis
            mid_perp = mid_perp / _safe_norm(mid_perp)

            pole_offset = self.pole_point - root_head
            pole_perp = pole_offset - jnp.dot(pole_offset, chain_axis) * chain_axis
            pole_perp = pole_perp / _safe_norm(pole_perp)

            cos_th = jnp.dot(mid_perp, pole_perp)
            return _safe_arccos(cos_th) ** 2 * self.weight

    _custom_objective_classes = {"LiveAimObj": LiveAimObj, "PoleTargetObj": PoleTargetObj}
    return _custom_objective_classes


# --------------------------------------------------------------------------
# Skeleton / FK construction
# --------------------------------------------------------------------------


def _mat_to_np4(m: mathutils.Matrix) -> np.ndarray:
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)


def build_skeleton_dict(armature_obj: bpy.types.Object) -> dict:
    """Build the same {name: {local_transform, children, parent,
    bone_length}} shape jax_ik.helper.load_skeleton_from_gltf produces, but
    directly from the armature's rest pose -- no file export round-trip,
    and no dependency on a particular root bone name.
    """
    skeleton = {}
    for bone in armature_obj.data.bones:
        if bone.parent is not None:
            local = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            local = bone.matrix_local.copy()
        skeleton[bone.name] = {
            "name": bone.name,
            "local_transform": _mat_to_np4(local),
            "children": [c.name for c in bone.children],
            "bone_length": float(bone.length) if bone.length > 1e-6 else 0.01,
            "parent": bone.parent.name if bone.parent else None,
        }
    return skeleton


def build_fk_solver(skeleton: dict, controlled_bones: list, bones_of_interest=None):
    """Construct an FKSolver from an already-built skeleton dict, bypassing
    FKSolver.__init__ (which only loads from a GLB/GLTF/URDF file on disk).
    Mirrors what FKSolver._pruned_view does internally in jax_ik.ik.
    """
    jax_ik_ik, _ = _ensure_jax_ik_imported()
    FKSolver = jax_ik_ik.FKSolver

    fk_solver = FKSolver.__new__(FKSolver)
    fk_solver.model_file = None
    fk_solver.file_type = ""
    fk_solver.limits = {}
    fk_solver.mesh_data = None
    fk_solver.sdf = None
    fk_solver.skeleton = skeleton

    keep_bones = None
    if bones_of_interest:
        keep_bones = fk_solver._bone_closure(controlled_bones, bones_of_interest)
    fk_solver._finish_init(controlled_bones, keep_bones)
    return fk_solver


def _skeleton_signature(skeleton: dict) -> tuple:
    """Cheap, content-based fingerprint of everything about `skeleton` that
    affects FK: per-bone parent, length and local_transform. Included in
    _fk_solver_cache's key so that editing an armature's rest pose or bone
    structure (e.g. resizing/reparenting bones in Edit Mode) automatically
    invalidates the cached FKSolver instead of silently reusing one built
    from the skeleton as it was before the edit -- relying on every call
    site that could change the armature to remember to call
    clear_fk_solver_cache() is exactly the kind of thing that's easy to
    miss (Edit Mode changes go through Blender's own tools, not this
    add-on's operators, so there is no natural hook to call it from).
    """
    return tuple(
        (
            name,
            data["parent"],
            round(data["bone_length"], 6),
            tuple(round(float(v), 6) for v in data["local_transform"].flat),
        )
        for name, data in sorted(skeleton.items())
    )


def get_fk_solver_cached(armature_obj, controlled_bones: list, bones_of_interest):
    skeleton = build_skeleton_dict(armature_obj)
    key = (
        armature_obj.name,
        tuple(controlled_bones),
        frozenset(bones_of_interest or ()),
        _skeleton_signature(skeleton),
    )
    fk_solver = _fk_solver_cache.get(key)
    if fk_solver is None:
        fk_solver = build_fk_solver(skeleton, controlled_bones, bones_of_interest)
        _fk_solver_cache[key] = fk_solver
        # Drop this armature's other entries (stale controlled_bones/
        # bones_of_interest/skeleton combinations) so editing back and
        # forth doesn't grow the cache without bound.
        for stale_key in [
            k for k in _fk_solver_cache if k[0] == armature_obj.name and k != key
        ]:
            del _fk_solver_cache[stale_key]
    return fk_solver


def clear_fk_solver_cache():
    """Drop every cached FKSolver. get_fk_solver_cached already invalidates
    itself automatically on any skeleton/topology change (see
    _skeleton_signature), so this is only needed for a full manual reset.
    """
    _fk_solver_cache.clear()


# --------------------------------------------------------------------------
# Chain / bounds / pose helpers
# --------------------------------------------------------------------------


def get_controlled_bones(armature_obj: bpy.types.Object, chain) -> list:
    """Root-to-tip ordered bone name list: chain.tip_bone plus up to
    chain.chain_length ancestors (0 = walk all the way to the root),
    matching the README's controlled_bones convention.
    """
    bones = armature_obj.data.bones
    tip = bones.get(chain.tip_bone)
    if tip is None:
        raise ValueError(f"Tip bone '{chain.tip_bone}' not found on armature")

    max_count = chain.chain_length if chain.chain_length > 0 else len(bones) + 1
    result = []
    b = tip
    count = 0
    while b is not None and count < max_count:
        result.append(b.name)
        b = b.parent
        count += 1
    result.reverse()
    return result


def sync_chain_bone_snapshots(armature_obj: bpy.types.Object, chain) -> None:
    """Keep chain.bone_snapshots in sync with chain's current controlled-bone
    set (chain.bone_snapshots doubles as "the controlled-bone set as of the
    last sync", so no separate cache is needed). Called from the tip_bone/
    chain_length property update callbacks in properties.py, i.e. every time
    a chain's topology changes -- including the moment a chain is first
    created, since setting chain.tip_bone fires the same callback.

    A bone newly entering control gets its current rotation captured as a
    restore point. A bone newly leaving control (chain shrank, or tip_bone
    moved to a different branch) has that captured rotation written straight
    back before the entry is dropped -- otherwise it would keep whatever
    rotation this chain's *last solve* left it at, and
    set_default_rotations_from_current_pose freezes every uncontrolled bone
    at its currently-evaluated pose for the *next* solve's FK baseline, so a
    stale solved rotation on a now-unaffected bone would silently corrupt
    that next solve's frame of reference.
    """
    try:
        new_bones = set(get_controlled_bones(armature_obj, chain))
    except ValueError:
        new_bones = set()

    old_bones = {s.bone_name for s in chain.bone_snapshots}
    pose_bones = armature_obj.pose.bones

    for name in new_bones - old_bones:
        pb = pose_bones.get(name)
        if pb is None:
            continue
        snap = chain.bone_snapshots.add()
        snap.bone_name = name
        snap.rotation_mode = pb.rotation_mode
        snap.rotation_euler = pb.rotation_euler
        snap.rotation_quaternion = pb.rotation_quaternion
        snap.rotation_axis_angle = pb.rotation_axis_angle

    for i in reversed(range(len(chain.bone_snapshots))):
        snap = chain.bone_snapshots[i]
        if snap.bone_name in new_bones:
            continue
        pb = pose_bones.get(snap.bone_name)
        if pb is not None:
            pb.rotation_mode = snap.rotation_mode
            pb.rotation_euler = snap.rotation_euler
            pb.rotation_quaternion = snap.rotation_quaternion
            pb.rotation_axis_angle = snap.rotation_axis_angle
        chain.bone_snapshots.remove(i)


def _bone_pose_delta(bone: bpy.types.Bone, pose_bones) -> mathutils.Matrix:
    """The local (relative-to-parent, relative-to-rest) rotation delta that
    reproduces this bone's *currently depsgraph-evaluated* pose (i.e. after
    whatever constraints/drivers/other IK are already doing to it), in the
    same "delta on top of local_array" space jax_ik's FK expects. This is
    what makes JAX-IK "adhere to existing constraints" for everything
    outside the controlled chain: uncontrolled bones are frozen at this
    delta for the duration of one solve.
    """
    if bone.parent is not None:
        rest_local = bone.parent.matrix_local.inverted() @ bone.matrix_local
        cur_local = pose_bones[bone.parent.name].matrix.inverted() @ pose_bones[bone.name].matrix
    else:
        rest_local = bone.matrix_local
        cur_local = pose_bones[bone.name].matrix
    return rest_local.inverted() @ cur_local


def set_default_rotations_from_current_pose(fk_solver, armature_obj: bpy.types.Object) -> None:
    """Overwrite FKSolver.default_rotations (identity by construction) with
    a snapshot of every bone's real, currently-evaluated local delta. Bones
    that end up controlled have their slot overwritten again by the
    optimizer itself, so what's written here for them doesn't matter.
    """
    import jax.numpy as jnp

    pose_bones = armature_obj.pose.bones
    bones = armature_obj.data.bones
    mats = [
        _mat_to_np4(_bone_pose_delta(bones[name], pose_bones)) for name in fk_solver.bone_names
    ]
    fk_solver.default_rotations = jnp.asarray(np.stack(mats, axis=0), dtype=jnp.float32)


_IK_AXIS_FIELDS = (
    ("lock_ik_x", "use_ik_limit_x", "ik_min_x", "ik_max_x"),
    ("lock_ik_y", "use_ik_limit_y", "ik_min_y", "ik_max_y"),
    ("lock_ik_z", "use_ik_limit_z", "ik_min_z", "ik_max_z"),
)


def compute_bounds(armature_obj: bpy.types.Object, controlled_bones: list, current_angles):
    """Per-axis (lower, upper) bounds in radians for every controlled bone,
    read from Blender's own IK-limit pose-bone fields -- the exact fields
    Blender's native IK constraint reads -- so joint limits set up for the
    native IK constraint are respected here too. A locked axis is pinned to
    its current angle rather than 0, so it doesn't jump when solving starts.
    """
    pose_bones = armature_obj.pose.bones
    lower, upper = [], []
    for i, name in enumerate(controlled_bones):
        pb = pose_bones[name]
        cur = current_angles[i * 3 : i * 3 + 3]
        for axis, (lock_attr, use_limit_attr, min_attr, max_attr) in enumerate(_IK_AXIS_FIELDS):
            if getattr(pb, lock_attr):
                v = float(cur[axis])
                lower.append(v)
                upper.append(v)
            elif getattr(pb, use_limit_attr):
                lower.append(float(getattr(pb, min_attr)))
                upper.append(float(getattr(pb, max_attr)))
            else:
                lower.append(math.radians(-180.0))
                upper.append(math.radians(180.0))
    return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)


def current_controlled_angles(armature_obj: bpy.types.Object, controlled_bones: list) -> np.ndarray:
    """Current joint-angle vector (Euler XYZ, jax_ik's convention) for the
    controlled bones, used both to seed the optimizer and as a fallback
    value for locked axes.
    """
    jax_ik_ik, _ = _ensure_jax_ik_imported()
    pose_bones = armature_obj.pose.bones
    bones = armature_obj.data.bones
    out = []
    for name in controlled_bones:
        delta = _bone_pose_delta(bones[name], pose_bones)
        a = jax_ik_ik.matrix_to_euler_xyz(np.array(delta.to_3x3(), dtype=np.float32))
        out.extend([float(a[0]), float(a[1]), float(a[2])])
    return np.array(out, dtype=np.float32)


def world_to_armature_local(armature_obj: bpy.types.Object, world_point: mathutils.Vector) -> np.ndarray:
    p = armature_obj.matrix_world.inverted() @ world_point
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def apply_result_to_pose(armature_obj: bpy.types.Object, controlled_bones: list, angle_vector) -> None:
    pose_bones = armature_obj.pose.bones
    for i, name in enumerate(controlled_bones):
        pb = pose_bones[name]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (
            float(angle_vector[i * 3 + 0]),
            float(angle_vector[i * 3 + 1]),
            float(angle_vector[i * 3 + 2]),
        )


def keyframe_chain(armature_obj: bpy.types.Object, chain, frame: int) -> None:
    """Insert a rotation_euler keyframe at `frame` for every bone this chain
    controls, reflecting whatever solve_chain just applied to the pose.
    """
    controlled_bones = get_controlled_bones(armature_obj, chain)
    pose_bones = armature_obj.pose.bones
    for name in controlled_bones:
        pose_bones[name].keyframe_insert(data_path="rotation_euler", frame=frame)


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


_KNOWN_OBJ_TYPES = (
    "DISTANCE",
    "LOOK_AT",
    "AVOID_SPHERE",
    "POLE_TARGET",
    "ZERO_ROTATION",
    "PREFER_CURRENT",
)


def build_objectives(armature_obj: bpy.types.Object, chain, controlled_bones: list, initial_angles) -> tuple:
    """Translate chain.objectives rows into jax_ik objective instances,
    split by each row's `optional` flag into (mandatory, optional).

    Mandatory objectives (optional=False, the default) must converge below
    chain.threshold for solve_chain's solve to count as successful; optional
    objectives are optimized every step right alongside them (so they still
    shape the pose) but never block that success, and never get reported as
    the success/failure value. See jax_ik.ik.solve_ik's docstring for the
    underlying mechanism.

    Returns (mandatory_objectives, optional_objectives, skip_reasons) --
    skip_reasons is a human-readable explanation for every *enabled* row
    that didn't turn into an objective, so callers can tell "nothing to
    solve" apart from "nothing is even configured yet" instead of failing
    silently either way.
    """
    _, obj_mod = _ensure_jax_ik_imported()
    custom = _get_custom_objectives()

    mandatory_objectives = []
    optional_objectives = []
    skip_reasons = []
    for i, item in enumerate(chain.objectives):
        if not item.enabled:
            continue

        is_optional = getattr(item, "optional", False)
        label = "optional" if is_optional else "mandatory"
        objectives = optional_objectives if is_optional else mandatory_objectives

        if item.obj_type not in _KNOWN_OBJ_TYPES:
            # Most likely leftover data from before an add-on update changed
            # the objective type list (Blender enums are stored positionally,
            # so an old entry can silently point at the wrong -- or no --
            # type after the list changes). Remove and re-add it.
            skip_reasons.append(
                f"{label} objective #{i + 1}: unrecognized type '{item.obj_type}' -- remove and re-add it"
            )
            continue

        if item.obj_type in ("DISTANCE", "LOOK_AT", "AVOID_SPHERE", "POLE_TARGET") and item.target_object is None:
            skip_reasons.append(f"{label} objective #{i + 1} ({item.obj_type}): no Target object set")
            continue

        if item.obj_type == "DISTANCE":
            bone_name = item.bone_name if item.bone_name else chain.tip_bone
            tp = world_to_armature_local(armature_obj, item.target_object.matrix_world.translation)
            objectives.append(
                obj_mod.DistanceObjTraj(
                    bone_name=bone_name, target_points=[tp], use_head=item.use_head, weight=item.weight
                )
            )
        elif item.obj_type == "LOOK_AT":
            bone_name = item.bone_name if item.bone_name else chain.tip_bone
            tp = world_to_armature_local(armature_obj, item.target_object.matrix_world.translation)
            objectives.append(
                custom["LiveAimObj"](
                    bone_name=bone_name, use_head=item.use_head, target_point=tp, weight=item.weight
                )
            )
        elif item.obj_type == "AVOID_SPHERE":
            center = world_to_armature_local(armature_obj, item.target_object.matrix_world.translation)
            objectives.append(
                obj_mod.SphereCollisionPenaltyObjTraj(
                    sphere_collider={"center": center, "radius": float(item.avoid_radius)},
                    weight=item.weight,
                )
            )
        elif item.obj_type == "POLE_TARGET":
            root_name = controlled_bones[0]
            tip_name = controlled_bones[-1]
            mid_name = item.bone_name if item.bone_name else controlled_bones[len(controlled_bones) // 2]
            if mid_name in (root_name, tip_name):
                skip_reasons.append(
                    f"{label} objective #{i + 1} (Pole Target): resolved bend joint '{mid_name}' is the "
                    "chain's root/tip -- needs a chain of 3+ bones, or set Bend Joint explicitly"
                )
                continue
            pole_pt = world_to_armature_local(armature_obj, item.target_object.matrix_world.translation)
            objectives.append(
                custom["PoleTargetObj"](
                    root_bone=root_name,
                    mid_bone=mid_name,
                    tip_bone=tip_name,
                    pole_point=pole_pt,
                    weight=item.weight,
                )
            )
        elif item.obj_type == "ZERO_ROTATION":
            objectives.append(obj_mod.BoneZeroRotationObj(weight=item.weight))
        elif item.obj_type == "PREFER_CURRENT":
            objectives.append(
                obj_mod.InitPoseObj(init_rot=initial_angles, last_position=True, weight=item.weight)
            )

    return mandatory_objectives, optional_objectives, skip_reasons


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def solve_chain(armature_obj: bpy.types.Object, chain):
    """Solve one chain once and apply the result to the pose. Used by both
    the manual Solve operator and the live-update depsgraph handler, so
    there is exactly one code path from "chain + Blender state" to
    "posed bones".

    Each row in chain.objectives is either Mandatory (the default) or
    Optional (its `optional` flag). Mandatory objectives must converge below
    chain.threshold for the solve to be considered successful; optional
    objectives are optimized every step right alongside them -- so they
    still shape the pose -- but never block that success. See
    jax_ik.ik.solve_ik's docstring for the underlying mechanism.

    Returns (steps, best_objective_value, message). message is "" on a
    normal solve; when best_objective_value is None, nothing was solved and
    message explains why (empty objectives list, no target set, tip bone
    missing, unrecognized/stale objective type, ...) instead of failing
    silently -- both the Solve button and Live rely on this to actually be
    visible to the user rather than just a console print or an
    easy-to-miss status-bar flash. When not None, best_objective_value is
    the best mandatory-objectives loss seen (or best combined loss, if this
    chain has no mandatory objectives) -- compare it against chain.threshold
    to tell whether the solve actually succeeded.
    """
    jax_ik_ik, obj_mod = _ensure_jax_ik_imported()

    if not chain.tip_bone:
        return 0, None, "Tip Bone is not set."

    controlled_bones = get_controlled_bones(armature_obj, chain)
    initial_angles = current_controlled_angles(armature_obj, controlled_bones)

    mandatory_objectives, optional_objectives, skip_reasons = build_objectives(
        armature_obj, chain, controlled_bones, initial_angles
    )

    if not mandatory_objectives and not optional_objectives:
        if not chain.objectives:
            return 0, None, "No objectives added yet."
        if not skip_reasons:
            return 0, None, "No enabled objectives."
        return 0, None, "Nothing to solve -- " + "; ".join(skip_reasons)

    # Two small, always-on regularizers, invisible in the UI -- chain.objectives
    # never contains them, they're added here unconditionally on every solve:
    # a tiny pull toward the rest pose (BoneZeroRotationObj) and toward
    # wherever the pose already was when this solve started (InitPoseObj).
    # Weighted low enough (0.05) to be negligible next to any real objective,
    # but they noticeably stabilize the optimizer -- without them, an
    # under-constrained joint (e.g. one axis a Reach Target doesn't pin down)
    # is free to drift to an arbitrary angle; these bias it back toward the
    # smallest, least surprising change instead. Mandatory (not optional) in
    # solve_ik's sense: their residual is folded into the mandatory-loss
    # convergence/threshold check right alongside the user's own mandatory
    # objectives, so a solve only registers as successful once both the
    # user's targets *and* these stabilizers have settled.
    auto_regularizers = [
        obj_mod.BoneZeroRotationObj(weight=0.05),
        obj_mod.InitPoseObj(init_rot=initial_angles, last_position=True, weight=0.05),
    ]
    mandatory_objectives = mandatory_objectives + auto_regularizers

    all_objectives = mandatory_objectives + optional_objectives

    # Mirrors InverseKinematicsSolver._fk_solver_for: if anything needs the
    # full, unpruned skeleton (referenced_bones() -> None, e.g. Avoid
    # Sphere, which has to check every bone segment), use it; otherwise
    # prune FK down to just the bones actually referenced.
    referenced = [fn.referenced_bones() for fn in all_objectives]
    if any(r is None for r in referenced):
        bones_of_interest = None
    else:
        bones_of_interest = set()
        for r in referenced:
            bones_of_interest.update(r)

    fk_solver = get_fk_solver_cached(armature_obj, controlled_bones, bones_of_interest)
    set_default_rotations_from_current_pose(fk_solver, armature_obj)

    lower, upper = compute_bounds(armature_obj, controlled_bones, initial_angles)

    # Single fixed prefix frame (the pose as it is now) + one free frame to
    # solve, matching how InverseKinematicsSolver.solve() itself always
    # builds a >=2-frame trajectory before calling solve_ik -- solve_ik's
    # own 1D-input default-mask path doesn't produce a correctly shaped
    # mask, so an explicit 2-frame trajectory + mask sidesteps that.
    #
    # A longer, real-solved-history trajectory + a CombinedDerivativeObj
    # smoothness term was tried here (extending the prefix to the last few
    # actually-solved frames, to smooth a reconfiguration -- e.g. a Reach
    # Target dragged near the chain's own root, forcing a different-but-
    # equally-valid bend direction -- across several frames instead of one
    # abrupt pop). Reverted: BoneZeroRotationObj/InitPoseObj-style
    # objectives that jnp.mean() over *every* frame in the trajectory
    # (see objectives.ObjectiveFunction subclasses that don't restrict
    # themselves to the last frame) get their effective per-solve gradient
    # diluted by however many extra fixed frames are added -- an N-frame
    # prefix instead of a 1-frame one silently weakens Zero Rotation to
    # 1/N of its configured strength, which is what made the first version
    # of this fix look like an improvement (a weaker regularizer let Reach
    # track more precisely) while actually increasing frame-to-frame
    # jitter (jump std/p95 measurably worse) since the redundant chain was
    # left less constrained. Compensating that dilution (scaling Zero
    # Rotation's weight by trajectory length) removes the regression, but
    # then the smoothness term's own effect on jitter is flat-to-noisy
    # across every history-length/weight tried on paper_evaluation/
    # ik_jax_lib_bench_drag.py, with no robust improvement region -- see
    # that benchmark and its git history for the numbers. Not worth the
    # added complexity (a pose-history cache, another Optional objective)
    # for an effect this unreliable.
    init_rot = np.stack([initial_angles, initial_angles], axis=0)
    mask = np.array([False, True])

    steps, final_traj, best_obj, _ = jax_ik_ik.solve_ik(
        init_rot=init_rot,
        lower_bounds=lower,
        upper_bounds=upper,
        mandatory_obj_fns=tuple(mandatory_objectives),
        optional_obj_fns=tuple(optional_objectives),
        fksolver=fk_solver,
        threshold=chain.threshold,
        num_steps=chain.num_steps,
        learning_rate=chain.learning_rate,
        patience=chain.patience,
        mask=mask,
    )

    final_angles = np.asarray(final_traj)[-1]
    apply_result_to_pose(armature_obj, controlled_bones, final_angles)

    message = ""
    if skip_reasons:
        message = "Some objectives were skipped -- " + "; ".join(skip_reasons)
    return int(steps), float(best_obj), message
