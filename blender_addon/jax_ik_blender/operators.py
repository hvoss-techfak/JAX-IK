import bpy
import numpy as np

from . import bridge, smoothing


def _active_armature(context):
    obj = context.object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    return None


class JAXIK_OT_add_chain(bpy.types.Operator):
    bl_idname = "jaxik.add_chain"
    bl_label = "New JAX-IK Chain"
    bl_description = "Create a new JAX-IK chain (pick its Tip Bone in the panel afterward)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        chain = chains.add()
        active_bone = context.active_pose_bone
        chain.tip_bone = active_bone.name if active_bone is not None else ""
        chain.name = f"JAX-IK: {chain.tip_bone}" if chain.tip_bone else "New JAX-IK Chain"
        armature_obj.data.active_jax_ik_chain_index = len(chains) - 1

        # Zero Rotation / Prefer Current Pose are added automatically by
        # bridge.solve_chain itself on every solve (always-on, invisible --
        # not stored here as chain.objectives rows), so nothing to add here.

        bridge.clear_fk_solver_cache()
        return {"FINISHED"}


class JAXIK_OT_remove_chain(bpy.types.Operator):
    bl_idname = "jaxik.remove_chain"
    bl_label = "Remove JAX-IK Chain"
    bl_description = "Delete this JAX-IK chain and all of its objectives"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if 0 <= self.chain_index < len(chains):
            chains.remove(self.chain_index)
            armature_obj.data.active_jax_ik_chain_index = max(0, self.chain_index - 1)
            bridge.clear_fk_solver_cache()
        return {"FINISHED"}


class JAXIK_OT_add_objective(bpy.types.Operator):
    bl_idname = "jaxik.add_objective"
    bl_label = "Add Objective"
    bl_description = "Add a new objective to this chain (Mandatory by default -- toggle Optional in its details)"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            return {"CANCELLED"}
        chain = chains[self.chain_index]
        item = chain.objectives.add()
        chain.active_objective_index = len(chain.objectives) - 1
        item.bone_name = chain.tip_bone
        return {"FINISHED"}


class JAXIK_OT_remove_objective(bpy.types.Operator):
    bl_idname = "jaxik.remove_objective"
    bl_label = "Remove Objective"
    bl_description = "Remove the selected objective from this chain"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()
    objective_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            return {"CANCELLED"}
        chain = chains[self.chain_index]
        if 0 <= self.objective_index < len(chain.objectives):
            chain.objectives.remove(self.objective_index)
            chain.active_objective_index = max(0, self.objective_index - 1)
        return {"FINISHED"}


class JAXIK_OT_move_objective(bpy.types.Operator):
    bl_idname = "jaxik.move_objective"
    bl_label = "Move Objective"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()
    direction: bpy.props.EnumProperty(items=(("UP", "Up", ""), ("DOWN", "Down", "")))

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    @classmethod
    def description(cls, context, properties):
        way = "up" if properties.direction == "UP" else "down"
        return f"Move this objective {way} in the list -- changes display order only, not solve behavior"

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            return {"CANCELLED"}
        chain = chains[self.chain_index]
        items = chain.objectives
        idx = chain.active_objective_index
        new_idx = idx - 1 if self.direction == "UP" else idx + 1
        if not (0 <= idx < len(items)) or not (0 <= new_idx < len(items)):
            return {"CANCELLED"}
        items.move(idx, new_idx)
        chain.active_objective_index = new_idx
        return {"FINISHED"}


class JAXIK_OT_info(bpy.types.Operator):
    """Inline hover-tooltip for a plain section label that isn't itself a
    property or operator. Always disabled (poll returns False) -- Blender
    still shows a disabled button's tooltip on hover, which is the whole
    point; execute() is unreachable.
    """

    bl_idname = "jaxik.info"
    bl_label = ""
    bl_options = {"INTERNAL"}

    message: bpy.props.StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return False

    @classmethod
    def description(cls, context, properties):
        return properties.message

    def execute(self, context):
        return {"CANCELLED"}


def _solve_status_text(chain, steps: int, obj_value: float) -> str:
    """Phrase a completed solve's status in terms of success against
    chain.threshold rather than just echoing the raw loss number. Every
    solve has at least the two hidden, always-mandatory stabilizers (Zero
    Rotation, Prefer Current Pose) on top of whatever's in chain.objectives
    (see bridge.solve_chain), so obj_value is always a "best mandatory loss"
    in solve_ik's sense -- "below threshold" genuinely means "everything
    mandatory (the user's own objectives *and* the stabilizers) settled".
    """
    if obj_value <= chain.threshold:
        return f"Solved in {steps} steps -- target reached (loss={obj_value:.5f})"
    return f"Solved in {steps} steps -- target not fully reached (loss={obj_value:.5f}, threshold={chain.threshold:.5f})"


class JAXIK_OT_solve(bpy.types.Operator):
    bl_idname = "jaxik.solve"
    bl_label = "Solve"
    bl_description = "Run the JAX-IK solver for this chain and apply the result to the pose"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            self.report({"ERROR"}, "No such JAX-IK chain")
            return {"CANCELLED"}
        chain = chains[self.chain_index]
        try:
            steps, obj_value, message = bridge.solve_chain(armature_obj, chain)
        except Exception as exc:  # noqa: BLE001 - surface any solver failure to the user
            chain.last_status = f"Solve failed: {exc}"
            chain.last_status_is_error = True
            self.report({"ERROR"}, chain.last_status)
            return {"CANCELLED"}

        if obj_value is None:
            chain.last_status = f"Nothing to solve: {message}"
            chain.last_status_is_error = True
            self.report({"WARNING"}, chain.last_status)
        else:
            chain.last_status = _solve_status_text(chain, steps, obj_value) + (
                f" -- {message}" if message else ""
            )
            chain.last_status_is_error = False
            self.report({"INFO"}, chain.last_status)
        return {"FINISHED"}


class JAXIK_OT_use_playback_range(bpy.types.Operator):
    bl_idname = "jaxik.use_playback_range"
    bl_label = "Use Playback Range"
    bl_description = "Set Start/End Frame to the scene's playback range"
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            return {"CANCELLED"}
        chain = chains[self.chain_index]
        chain.bake_start_frame = context.scene.frame_start
        chain.bake_end_frame = context.scene.frame_end
        return {"FINISHED"}


class JAXIK_OT_bake(bpy.types.Operator):
    bl_idname = "jaxik.bake"
    bl_label = "Bake to Keyframes"
    bl_description = (
        "Solve this chain frame by frame across Start Frame..End Frame (End < Start bakes "
        "backward) and keyframe the result on every controlled bone. If Autosmooth is on, "
        "smooths the solved rotations across frames before keyframing, to reduce flicker"
    )
    bl_options = {"REGISTER", "UNDO"}

    chain_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        armature_obj = _active_armature(context)
        chains = armature_obj.data.jax_ik_chains
        if not (0 <= self.chain_index < len(chains)):
            self.report({"ERROR"}, "No such JAX-IK chain")
            return {"CANCELLED"}
        chain = chains[self.chain_index]

        start, end = chain.bake_start_frame, chain.bake_end_frame
        step = 1 if end >= start else -1
        frames = list(range(start, end + step, step))

        scene = context.scene
        original_frame = scene.frame_current
        wm = context.window_manager
        wm.progress_begin(0, len(frames))

        autosmooth = chain.bake_autosmooth
        controlled_bones = bridge.get_controlled_bones(armature_obj, chain)
        # Autosmooth needs every frame's raw result before it can smooth
        # any of them, so it collects angles here instead of keyframing
        # per-frame -- keyframing (all frames at once) only happens after
        # the loop below, once smoothing has run. This does mean a bake
        # that fails partway through with Autosmooth on writes *no*
        # keyframes at all, unlike a plain bake (which keeps whatever it
        # keyframed before the failure): smoothing a partial range isn't
        # meaningful, so there is nothing sensible to keep.
        solved_angles = [] if autosmooth else None

        solved = 0
        try:
            for i, frame in enumerate(frames):
                scene.frame_set(frame)
                try:
                    steps, obj_value, message = bridge.solve_chain(armature_obj, chain)
                except Exception as exc:  # noqa: BLE001 - stop the bake, report where/why
                    chain.last_status = f"Bake stopped at frame {frame}: {exc}"
                    chain.last_status_is_error = True
                    self.report({"ERROR"}, chain.last_status)
                    return {"CANCELLED"}
                if obj_value is None:
                    chain.last_status = f"Bake stopped at frame {frame}: {message}"
                    chain.last_status_is_error = True
                    self.report({"ERROR"}, chain.last_status)
                    return {"CANCELLED"}

                if autosmooth:
                    # apply_result_to_pose (inside solve_chain) only sets
                    # rotation_euler -- it doesn't force Blender to
                    # re-evaluate pose_bone.matrix from that new value, so
                    # reading current_controlled_angles right away would
                    # return whatever matrix was last evaluated (i.e. the
                    # pose from *before* this frame's solve -- the original,
                    # unsolved pose on the very first frame). Force the
                    # re-evaluation so the angles collected for smoothing
                    # are actually this frame's solved result.
                    context.view_layer.update()
                    solved_angles.append(bridge.current_controlled_angles(armature_obj, controlled_bones))
                else:
                    bridge.keyframe_chain(armature_obj, chain, frame)
                solved += 1
                wm.progress_update(i)

            if autosmooth:
                smoothed = smoothing.smooth_angle_sequence(
                    np.stack(solved_angles, axis=0), chain.bake_autosmooth_amount
                )
                for i, frame in enumerate(frames):
                    bridge.apply_result_to_pose(armature_obj, controlled_bones, smoothed[i])
                    bridge.keyframe_chain(armature_obj, chain, frame)
        finally:
            wm.progress_end()
            scene.frame_set(original_frame)

        status = f"Baked {solved} frame(s) ({start} -> {end})"
        if autosmooth:
            status += f" -- smoothed (amount={chain.bake_autosmooth_amount:.2f})"
        chain.last_status = status
        chain.last_status_is_error = False
        self.report({"INFO"}, chain.last_status)
        return {"FINISHED"}


classes = (
    JAXIK_OT_add_chain,
    JAXIK_OT_remove_chain,
    JAXIK_OT_add_objective,
    JAXIK_OT_remove_objective,
    JAXIK_OT_move_objective,
    JAXIK_OT_info,
    JAXIK_OT_solve,
    JAXIK_OT_use_playback_range,
    JAXIK_OT_bake,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
