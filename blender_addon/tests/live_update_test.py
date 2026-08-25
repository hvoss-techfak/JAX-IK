"""Regression test for Live mode actually re-solving on scene changes --
not just "the handler didn't raise an exception" (lifecycle_test.py's
weaker check, which passed even when a real regression made Live silently
do nothing), but "the pose actually changed to track the moved target",
exactly what a user dragging the target Empty with Live enabled expects to
see happen.
"""

import sys

import bpy
import numpy as np

import addon_utils

addon_utils.enable("bl_ext.user_default.jax_ik_blender", default_set=True, persistent=True)
from bl_ext.user_default.jax_ik_blender import bridge, handlers

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


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
bpy.context.view_layer.objects.active = armature_obj
bpy.context.view_layer.update()

target = bpy.data.objects.new("LiveTarget", None)
bpy.context.collection.objects.link(target)
target.location = (0.3, 1.6, 0.2)
bpy.context.view_layer.update()

r = bpy.ops.jaxik.add_chain()
check("add_chain", r == {"FINISHED"}, str(r))
chain = armature_obj.data.jax_ik_chains[-1]
chain.tip_bone = "tip"
chain.chain_length = 2  # whole chain (root, tip) -- chain_length has a hard min of 1 now
chain.num_steps = 500
chain.threshold = 0.0005
chain.live_update = True
chain_index = len(armature_obj.data.jax_ik_chains) - 1

r = bpy.ops.jaxik.add_objective(chain_index=chain_index)
check("add_objective", r == {"FINISHED"}, str(r))
item = chain.objectives[-1]
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

# Baseline: before any solve, the tip sits at the rest pose, far from the
# target the objective was just pointed at.
tip_before = np.array(armature_obj.matrix_world @ armature_obj.pose.bones["tip"].tail)
dist_before = float(np.linalg.norm(tip_before - np.array(target.location)))
print("dist before any live tick:", dist_before)

# This is exactly what happens when the user drags the target Empty with
# Live enabled: something moves, Blender fires depsgraph_update_post, and
# the registered handler should pick it up and re-solve.
check(
    "live handler is actually registered",
    handlers._on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post,
)

target.location = (0.35, 1.55, 0.15)
bpy.context.view_layer.update()  # fires depsgraph_update_post

tip_after = np.array(armature_obj.matrix_world @ armature_obj.pose.bones["tip"].tail)
dist_after = float(np.linalg.norm(tip_after - np.array(target.location)))
print("dist after live tick:", dist_after, "chain.last_status:", repr(chain.last_status))

check("live tick actually solved (status set, no error)", chain.last_status != "" and not chain.last_status_is_error, chain.last_status)
check("live tick moved the bone toward the (new) target", dist_after < dist_before, f"before={dist_before:.4f} after={dist_after:.4f}")
check("live tick converged reasonably close", dist_after < 0.1, f"dist_after={dist_after:.4f}")

# A second, independent move should also be picked up (not just the first
# tick after enabling Live).
target.location = (-0.3, 1.5, 0.3)
bpy.context.view_layer.update()
tip_after2 = np.array(armature_obj.matrix_world @ armature_obj.pose.bones["tip"].tail)
dist_after2 = float(np.linalg.norm(tip_after2 - np.array(target.location)))
print("dist after second live tick:", dist_after2)
check("second live tick also solved", dist_after2 < 0.1, f"dist_after2={dist_after2:.4f}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
