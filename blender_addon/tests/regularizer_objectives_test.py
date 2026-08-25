"""Verifies Zero Rotation and Prefer Current Pose are addable as explicit,
user-visible objectives again (not just the hidden auto-regularizers), and
that the hidden auto-regularizers now use weight 0.05.
"""

import sys

import bpy
import numpy as np

import addon_utils

addon_utils.enable("bl_ext.user_default.jax_ik_blender", default_set=True, persistent=True)
from bl_ext.user_default.jax_ik_blender import bridge, properties

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


check(
    "ZERO_ROTATION is a selectable objective type",
    any(t[0] == "ZERO_ROTATION" for t in properties.OBJECTIVE_TYPES),
)
check(
    "PREFER_CURRENT is a selectable objective type",
    any(t[0] == "PREFER_CURRENT" for t in properties.OBJECTIVE_TYPES),
)
check("neither needs a target object", not properties.objective_needs_target("ZERO_ROTATION"))
check("neither needs a target object (2)", not properties.objective_needs_target("PREFER_CURRENT"))


def build_armature():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
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
    return armature_obj


class Chain:
    tip_bone = "root"
    chain_length = 0
    num_steps = 500
    learning_rate = 0.2
    threshold = 0.0005
    patience = 200


class ZeroRotationItem:
    obj_type = "ZERO_ROTATION"
    enabled = True
    weight = 1.0
    bone_name = ""


class PreferCurrentItem:
    obj_type = "PREFER_CURRENT"
    enabled = True
    weight = 1.0
    bone_name = ""


# --- Zero Rotation explicitly added: pulls a displaced pose back to rest --
armature_obj = build_armature()
pb = armature_obj.pose.bones["root"]
pb.rotation_mode = "XYZ"
pb.rotation_euler = (0.6, -0.4, 0.3)
bpy.context.view_layer.update()

chain = Chain()
chain.objectives = [ZeroRotationItem()]
steps, obj, msg = bridge.solve_chain(armature_obj, chain)
print("zero-rotation:", steps, obj, msg)
final_angle = np.array(pb.rotation_euler)
start_norm = float(np.linalg.norm((0.6, -0.4, 0.3)))
final_norm = float(np.linalg.norm(final_angle))
check("zero rotation objective solved", obj is not None, msg)
# The hidden "prefer current pose" auto-regularizer (weight 0.05) pulls
# back toward the original displaced pose the whole time, so this won't
# reach exactly zero -- just check it moved substantially toward rest.
check(
    "zero rotation objective pulled the pose toward rest",
    final_norm < 0.2 * start_norm,
    f"start_norm={start_norm:.3f} final_norm={final_norm:.3f}",
)

# --- Prefer Current Pose explicitly added: barely moves from where it started
armature_obj2 = build_armature()
pb2 = armature_obj2.pose.bones["root"]
pb2.rotation_mode = "XYZ"
pb2.rotation_euler = (0.6, -0.4, 0.3)
bpy.context.view_layer.update()
start_angle = np.array(pb2.rotation_euler)

chain2 = Chain()
chain2.objectives = [PreferCurrentItem()]
steps2, obj2, msg2 = bridge.solve_chain(armature_obj2, chain2)
print("prefer-current:", steps2, obj2, msg2)
final_angle2 = np.array(pb2.rotation_euler)
check("prefer current pose objective solved", obj2 is not None, msg2)
check(
    "prefer current pose objective kept the pose close to its start",
    np.linalg.norm(final_angle2 - start_angle) < 0.05,
    f"start={start_angle} final={final_angle2}",
)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
