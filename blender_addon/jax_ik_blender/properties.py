"""PropertyGroups for JAX-IK chains and objectives, stored on the Armature
data-block so they save with the .blend file like any other rig setup.
"""

import bpy

from . import bridge

OBJECTIVE_TYPES = (
    ("DISTANCE", "Reach Target", "Pull the bone's head/tail toward the target object"),
    (
        "LOOK_AT",
        "Look At",
        "Orient the bone's head-to-tail axis at the target object, without moving the tip onto it",
    ),
    (
        "POLE_TARGET",
        "Pole Target",
        "Control which way the chain bends (e.g. elbow/knee direction) toward the target object",
    ),
    (
        "AVOID_SPHERE",
        "Avoid Point",
        "Keep every bone in the armature clear of a sphere centered on the target object",
    ),
    (
        "ZERO_ROTATION",
        "Zero Rotation",
        "Regularizer: pull this chain's controlled joints toward their rest rotation",
    ),
    (
        "PREFER_CURRENT",
        "Prefer Current Pose",
        "Regularizer: minimize this chain's movement away from the pose at solve start",
    ),
)

_NEEDS_TARGET = {"DISTANCE", "LOOK_AT", "POLE_TARGET", "AVOID_SPHERE"}
_NEEDS_USE_HEAD = {"DISTANCE", "LOOK_AT"}
_NEEDS_BONE_NAME = {"DISTANCE", "LOOK_AT", "POLE_TARGET"}


def objective_needs_target(obj_type: str) -> bool:
    return obj_type in _NEEDS_TARGET


def objective_needs_use_head(obj_type: str) -> bool:
    return obj_type in _NEEDS_USE_HEAD


def objective_needs_bone_name(obj_type: str) -> bool:
    return obj_type in _NEEDS_BONE_NAME


_TYPE_LABELS = {t[0]: t[1] for t in OBJECTIVE_TYPES}


def objective_type_label(obj_type: str) -> str:
    return _TYPE_LABELS.get(obj_type, obj_type)


class JaxIKObjective(bpy.types.PropertyGroup):
    obj_type: bpy.props.EnumProperty(
        name="Type",
        items=OBJECTIVE_TYPES,
        default="DISTANCE",
    )
    enabled: bpy.props.BoolProperty(
        name="Enabled",
        description="Include this objective when solving. Disabled objectives are skipped entirely",
        default=True,
    )
    optional: bpy.props.BoolProperty(
        name="Optional",
        description="Optional objectives are optimized every step right alongside Mandatory "
        "ones (so they still shape the pose) but are never required to converge and never block "
        "a successful solve. Leave off (Mandatory) for objectives that must converge below the "
        "chain's Threshold for the solve to count as successful",
        default=False,
    )
    weight: bpy.props.FloatProperty(
        name="Weight",
        description="Relative strength of this objective against the other Mandatory (or "
        "Optional) objectives. Higher values are prioritized more strongly",
        default=1.0,
        min=0.0,
        soft_max=10.0,
    )
    target_object: bpy.props.PointerProperty(
        name="Target",
        description="Object (usually an Empty) this objective measures against, in world space",
        type=bpy.types.Object,
    )
    use_head: bpy.props.BoolProperty(
        name="Target = Head",
        description="Measure from this bone's head. Unchecked (default): measure from its tail "
        "-- the actual tip of the bone, usually what you want a Reach Target/Look At to aim for",
        default=False,
    )
    bone_name: bpy.props.StringProperty(
        name="Bone",
        description=(
            "Reach Target / Look At: which bone this applies to (defaults to the chain's tip "
            "bone if empty). Pole Target: which bone's joint should bend toward the target "
            "(defaults to the middle of the chain if empty)"
        ),
        default="",
    )
    avoid_radius: bpy.props.FloatProperty(
        name="Avoid Radius",
        description="Radius of the sphere around the target object that bones should stay clear of",
        default=0.1,
        min=0.0,
    )


class JaxIKBoneSnapshot(bpy.types.PropertyGroup):
    """One bone's rotation, captured by sync_chain_bone_snapshots() at the
    moment it entered a chain's controlled-bone set. Written straight back
    to the pose bone (and dropped) the moment that bone leaves the set --
    see bridge.sync_chain_bone_snapshots for why.
    """

    bone_name: bpy.props.StringProperty()
    rotation_mode: bpy.props.StringProperty()
    rotation_euler: bpy.props.FloatVectorProperty(size=3)
    rotation_quaternion: bpy.props.FloatVectorProperty(size=4, default=(1.0, 0.0, 0.0, 0.0))
    rotation_axis_angle: bpy.props.FloatVectorProperty(size=4, default=(0.0, 0.0, 1.0, 0.0))


def _on_chain_topology_changed(chain, context) -> None:
    """update= callback for tip_bone/chain_length: keeps chain.bone_snapshots
    in sync with whatever bones this chain now controls, resetting any bone
    that just dropped out of control back to its pre-chain rotation. See
    bridge.sync_chain_bone_snapshots for the actual bookkeeping.
    """
    obj = context.object
    if obj is None or obj.type != "ARMATURE":
        return
    bridge.sync_chain_bone_snapshots(obj, chain)


class JaxIKChain(bpy.types.PropertyGroup):
    show_expanded: bpy.props.BoolProperty(
        name="Show Expanded",
        description="Expand this chain's settings in the panel",
        default=True,
    )
    name: bpy.props.StringProperty(
        name="Name",
        description="Display name for this chain -- just a label, has no effect on solving",
        default="JAX-IK Chain",
    )
    tip_bone: bpy.props.StringProperty(
        name="Tip Bone",
        description="The bone at the far end of this chain -- the solver walks up the "
        "hierarchy from here",
        default="",
        update=_on_chain_topology_changed,
    )
    chain_length: bpy.props.IntProperty(
        name="Chain Length",
        description="Number of bones up the hierarchy from the tip bone this chain controls. "
        "Bones outside this range are left exactly as they were before this chain last touched "
        "them. To control all the way to the root, set this at least as high as the branch's "
        "actual bone count",
        default=1,
        min=1,
        soft_max=64,
        update=_on_chain_topology_changed,
    )
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    live_update: bpy.props.BoolProperty(
        name="Live",
        description="Re-solve automatically while target objects move. Off by default: every "
        "re-solve runs the full optimizer, which can be noticeably slower than Blender's native IK "
        "on complex chains",
        default=False,
    )
    num_steps: bpy.props.IntProperty(
        name="Max Steps",
        description="Upper bound on optimizer iterations per solve, if Threshold/Patience "
        "haven't already stopped it early",
        default=1000,
        min=1,
        max=20000,
    )
    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate",
        description="Adam optimizer step size. Higher moves faster per step but can overshoot "
        "or destabilize; lower is steadier but slower to converge",
        default=0.2,
        min=0.0001,
        max=5.0,
    )
    last_status: bpy.props.StringProperty(
        name="Last Status",
        description="Result of the most recent solve (also shown in the panel below Solve)",
        default="",
    )
    last_status_is_error: bpy.props.BoolProperty(default=False)
    threshold: bpy.props.FloatProperty(
        name="Threshold",
        description="Solve is considered successful once the combined loss of all enabled "
        "Mandatory objectives falls below this value. Optional objectives keep influencing the "
        "pose every step but never block success",
        default=0.0005,
        min=0.0,
        precision=5,
    )
    patience: bpy.props.IntProperty(
        name="Patience",
        description="Stop early if this many consecutive steps pass with no improvement, even "
        "if Threshold hasn't been reached yet",
        default=100,
        min=1,
    )

    bake_start_frame: bpy.props.IntProperty(
        name="Start Frame", default=1, description="First frame to solve and keyframe"
    )
    bake_end_frame: bpy.props.IntProperty(
        name="End Frame",
        default=250,
        description="Last frame to solve and keyframe. Lower than Start Frame bakes backward",
    )
    bake_autosmooth: bpy.props.BoolProperty(
        name="Autosmooth",
        description="After every frame in the range is solved, smooth the resulting rotations "
        "across frames before keyframing -- reduces flicker from the solver settling into a "
        "slightly different pose on nearby frames. Applied once per bake, not during Live",
        default=False,
    )
    bake_autosmooth_amount: bpy.props.FloatProperty(
        name="Amount",
        description="How much to smooth: 0 leaves the raw per-frame solve untouched, 1 applies "
        "the strongest smoothing (widest temporal window). Values in between blend smoothly",
        default=0.3,
        min=0.0,
        max=1.0,
    )

    objectives: bpy.props.CollectionProperty(type=JaxIKObjective)
    active_objective_index: bpy.props.IntProperty()

    bone_snapshots: bpy.props.CollectionProperty(type=JaxIKBoneSnapshot)


classes = (JaxIKObjective, JaxIKBoneSnapshot, JaxIKChain)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Armature.jax_ik_chains = bpy.props.CollectionProperty(type=JaxIKChain)
    bpy.types.Armature.active_jax_ik_chain_index = bpy.props.IntProperty()


def unregister():
    del bpy.types.Armature.active_jax_ik_chain_index
    del bpy.types.Armature.jax_ik_chains
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
