import sys

import bpy

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


import addon_utils

MOD = "bl_ext.user_default.jax_ik_blender"

result = addon_utils.enable(MOD, default_set=True, persistent=True)
check("enable", result is not None)

from bl_ext.user_default.jax_ik_blender import bridge, handlers

check(
    "depsgraph handler registered",
    handlers._on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post,
)

# --- live update tick: build a chain, enable Live, force a depsgraph update,
# confirm it re-solves without raising / without infinite recursion.
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
armature_obj = bpy.context.object
ebones = armature_obj.data.edit_bones
root = ebones[0]
root.name = "root"
root.head = (0, 0, 0)
root.tail = (0, 1, 0)
tip = ebones.new("tip")
tip.parent = root
tip.use_connect = True
tip.head = root.tail
tip.tail = (0, 2, 0)
bpy.ops.object.mode_set(mode="POSE")

target = bpy.data.objects.new("LiveTarget", None)
bpy.context.collection.objects.link(target)
target.location = (0.3, 1.5, 0.2)
bpy.context.view_layer.update()

chain = armature_obj.data.jax_ik_chains.add()
chain.tip_bone = "tip"
chain.chain_length = 0
chain.num_steps = 200
chain.live_update = True
item = chain.objectives.add()
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

try:
    target.location = (0.4, 1.4, 0.1)
    bpy.context.view_layer.update()  # triggers depsgraph_update_post
    check("live update tick did not raise", True)
except Exception as exc:  # noqa: BLE001
    check("live update tick did not raise", False, str(exc))

check("solving guard reset after tick", handlers._is_solving is False)

result = addon_utils.disable(MOD, default_set=True)
check(
    "depsgraph handler removed after disable",
    handlers._on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post,
)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
