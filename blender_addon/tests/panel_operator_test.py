"""Verifies the sidebar-driven workflow: adding a chain from Object Mode
(no active pose bone), setting its tip bone via the same field the panel
exposes (prop_search-backed StringProperty), then solving -- exactly the
path the JAX-IK 3D-viewport sidebar panel drives, without needing a display.
"""

import sys

import bpy

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


import addon_utils

addon_utils.enable("bl_ext.user_default.jax_ik_blender", default_set=True, persistent=True)
from bl_ext.user_default.jax_ik_blender import bridge

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
bpy.ops.object.mode_set(mode="OBJECT")  # deliberately NOT pose mode, no active pose bone

check("no active pose bone", bpy.context.active_pose_bone is None)
check("add_chain poll passes in Object Mode", bpy.ops.jaxik.add_chain.poll())

result = bpy.ops.jaxik.add_chain()
check("add_chain executed", result == {"FINISHED"}, str(result))

chain = armature_obj.data.jax_ik_chains[-1]
check("new chain has blank tip_bone", chain.tip_bone == "", repr(chain.tip_bone))

# This is exactly what ui.py's prop_search(chain, "tip_bone", ...) writes to.
chain.tip_bone = "tip"
chain.chain_length = 2  # whole chain (root, tip) -- chain_length has a hard min of 1 now
chain.num_steps = 500
chain.threshold = 0.0005

target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (0.3, 1.6, 0.2)
bpy.context.view_layer.update()

item = chain.objectives.add()
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

bpy.context.view_layer.objects.active = armature_obj
result = bpy.ops.jaxik.solve(chain_index=len(armature_obj.data.jax_ik_chains) - 1)
check("solve operator finished", result == {"FINISHED"}, str(result))

bpy.context.view_layer.update()
tip_tail = armature_obj.matrix_world @ armature_obj.pose.bones["tip"].tail
dist = (tip_tail - target.matrix_world.translation).length
# Loosened now that the auto-added Optional objectives pull harder (weight
# 0.05, up from 0.01) -- trading a bit of precision for stability is intentional.
check("tip reached target via sidebar-style flow", dist < 0.1, f"dist={dist:.4f}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
