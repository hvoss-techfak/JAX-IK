"""Regression test: Zero Rotation / Prefer Current Pose are hardcoded,
always-on, and invisible -- solve_chain must apply them on every solve
without ever adding rows to chain.objectives, and a chain with zero
configured objectives must still report "nothing to solve" rather than
silently succeeding off the hidden regularizers alone.
"""

import sys

import bpy

import addon_utils

addon_utils.enable("bl_ext.user_default.jax_ik_blender", default_set=True, persistent=True)
from bl_ext.user_default.jax_ik_blender import bridge

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
bpy.ops.object.mode_set(mode="POSE")
bpy.context.view_layer.objects.active = armature_obj
bpy.context.view_layer.update()

target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (0.3, 0.8, 0.2)
bpy.context.view_layer.update()

r = bpy.ops.jaxik.add_chain()
check("add_chain", r == {"FINISHED"}, str(r))
chain = armature_obj.data.jax_ik_chains[-1]
chain_index = len(armature_obj.data.jax_ik_chains) - 1
chain.tip_bone = "root"
chain.num_steps = 500
chain.threshold = 0.0005

# --- an empty chain must not silently "succeed" off the hidden pair --------
steps0, obj_value0, message0 = bridge.solve_chain(armature_obj, chain)
check("empty chain reports nothing to solve", obj_value0 is None, f"steps={steps0} obj={obj_value0} msg={message0}")
check("empty chain message names it", message0 == "No objectives added yet.", message0)

# --- one mandatory objective: hidden regularizers apply but stay invisible -
r = bpy.ops.jaxik.add_objective(chain_index=chain_index)
check("add_objective", r == {"FINISHED"}, str(r))
item = chain.objectives[-1]
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

count_before = len(chain.objectives)
steps, obj_value, message = bridge.solve_chain(armature_obj, chain)
count_after = len(chain.objectives)

check("solve with one mandatory objective succeeded", obj_value is not None, f"steps={steps} obj={obj_value} msg={message}")
check(
    "solving did not add any rows to chain.objectives (regularizers stay hidden)",
    count_after == count_before == 1,
    f"before={count_before} after={count_after}",
)
check(
    "no row's obj_type is ZERO_ROTATION/PREFER_CURRENT (never injected into the visible list)",
    all(o.obj_type not in ("ZERO_ROTATION", "PREFER_CURRENT") for o in chain.objectives),
    [o.obj_type for o in chain.objectives],
)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
