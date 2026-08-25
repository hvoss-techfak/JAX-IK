import bpy

from . import properties


class JAXIK_UL_objectives(bpy.types.UIList):
    bl_idname = "JAXIK_UL_objectives"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(
            item,
            "optional",
            text="",
            icon="UNPINNED" if item.optional else "PINNED",
            toggle=True,
        )
        label = properties.objective_type_label(item.obj_type)
        if item.target_object is not None:
            label += f"  ->  {item.target_object.name}"
        row.label(text=label, translate=False)


def _info_icon(layout, message):
    layout.operator("jaxik.info", text="", icon="QUESTION", emboss=False).message = message


def _draw_chain_body(layout, armature_obj, chain, chain_index):
    col = layout.column()

    col.prop_search(chain, "tip_bone", armature_obj.data, "bones", text="Tip Bone")

    col.prop(chain, "chain_length")

    row = col.row(align=True)
    row.operator("jaxik.solve", text="Solve", icon="PLAY").chain_index = chain_index
    row.prop(chain, "live_update", toggle=True, icon="RADIOBUT_ON" if chain.live_update else "RADIOBUT_OFF")

    if chain.last_status:
        col.label(
            text=chain.last_status,
            icon="ERROR" if chain.last_status_is_error else "INFO",
        )

    box = col.box()
    header = box.row(align=True)
    header.label(text="Objectives")
    _info_icon(
        header,
        "Each objective is either Mandatory (pin icon: must converge below Threshold, in "
        "Solver Settings, for the solve to count as successful) or Optional (unpinned: toggle "
        "the pin icon, or the Optional checkbox in the details below) -- optional objectives "
        "are optimized every step right alongside mandatory ones, so they still shape the pose, "
        "but are never required to converge and never block success. Every solve also quietly "
        "applies two built-in low-weight Mandatory stabilizers (pull toward rest / toward the "
        "pose the solve started from) on top of whatever's listed here -- not shown, always on, "
        "and part of what Threshold judges success against.",
    )

    row = box.row()
    row.template_list(
        "JAXIK_UL_objectives",
        "",
        chain,
        "objectives",
        chain,
        "active_objective_index",
        rows=3,
    )
    sub = row.column(align=True)
    sub.operator("jaxik.add_objective", text="", icon="ADD").chain_index = chain_index
    remove_op = sub.operator("jaxik.remove_objective", text="", icon="REMOVE")
    remove_op.chain_index = chain_index
    remove_op.objective_index = chain.active_objective_index
    sub.separator()
    up_op = sub.operator("jaxik.move_objective", text="", icon="TRIA_UP")
    up_op.chain_index = chain_index
    up_op.direction = "UP"
    down_op = sub.operator("jaxik.move_objective", text="", icon="TRIA_DOWN")
    down_op.chain_index = chain_index
    down_op.direction = "DOWN"

    if 0 <= chain.active_objective_index < len(chain.objectives):
        item = chain.objectives[chain.active_objective_index]
        detail = box.box()
        detail.prop(item, "obj_type")
        detail.prop(item, "optional")
        if properties.objective_needs_target(item.obj_type):
            detail.prop(item, "target_object")
        detail.prop(item, "weight")
        if properties.objective_needs_bone_name(item.obj_type):
            bone_label = "Bend Joint (blank = middle)" if item.obj_type == "POLE_TARGET" else "Bone (blank = tip)"
            detail.prop_search(item, "bone_name", armature_obj.data, "bones", text=bone_label)
        if properties.objective_needs_use_head(item.obj_type):
            detail.prop(item, "use_head")
        if item.obj_type == "AVOID_SPHERE":
            detail.prop(item, "avoid_radius")

    settings = col.box()
    settings_header = settings.row(align=True)
    settings_header.label(text="Solver Settings")
    _info_icon(
        settings_header,
        "Controls how the optimizer runs: Max Steps caps how long it can run, Learning Rate "
        "controls how big each step is, Threshold defines what counts as \"solved\" (compared "
        "against the Mandatory objectives' combined loss), and Patience stops early once that "
        "many steps pass with no improvement.",
    )
    settings.prop(chain, "num_steps")
    settings.prop(chain, "learning_rate")
    settings.prop(chain, "threshold")
    settings.prop(chain, "patience")

    bake = col.box()
    bake_header = bake.row(align=True)
    bake_header.label(text="Bake to Keyframes")
    _info_icon(
        bake_header,
        "Solves this chain frame by frame across Start Frame..End Frame and keyframes the "
        "result on every controlled bone -- for turning a Live-updated animation into normal "
        "keyframes you can edit afterward. Autosmooth, if enabled, smooths the solved rotation "
        "across frames (Amount: 0 = off, 1 = strongest) before keyframing, to reduce flicker "
        "from the solver settling into a slightly different pose on nearby frames.",
    )
    row = bake.row(align=True)
    row.prop(chain, "bake_start_frame")
    row.prop(chain, "bake_end_frame")
    bake.operator("jaxik.use_playback_range", text="Use Playback Range").chain_index = chain_index

    smooth_row = bake.row(align=True)
    smooth_row.prop(chain, "bake_autosmooth")
    amount_sub = smooth_row.row(align=True)
    amount_sub.enabled = chain.bake_autosmooth
    amount_sub.prop(chain, "bake_autosmooth_amount", slider=True)
    bake.operator("jaxik.bake", text="Bake to Keyframes", icon="KEY_HLT").chain_index = chain_index


class JAXIK_PT_view3d(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "JAX-IK"
    bl_label = "JAX-IK"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        layout = self.layout
        armature_obj = context.object
        chains = armature_obj.data.jax_ik_chains

        available, error = _dependency_status()
        if not available:
            box = layout.box()
            box.label(text="jax-ik is not available", icon="ERROR")
            box.label(text=error[:60])
            return

        layout.operator("jaxik.add_chain", text="New JAX-IK Chain", icon="ADD")

        if len(chains) == 0:
            layout.label(text="No JAX-IK chains on this armature yet.")
            return

        for i, chain in enumerate(chains):
            box = layout.box()
            header, panel = box.panel_prop(chain, "show_expanded")
            header.prop(chain, "name", text="")
            header.operator("jaxik.remove_chain", text="", icon="X").chain_index = i
            if panel:
                _draw_chain_body(panel, armature_obj, chain, i)


def _dependency_status():
    from . import bridge

    return bridge.is_available()


classes = (JAXIK_UL_objectives, JAXIK_PT_view3d)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
