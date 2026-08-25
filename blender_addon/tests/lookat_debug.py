"""Verifies the Look At fix: the bone should actually end up pointing at the
target (small geometric angle), not just converge to a low optimizer loss
that turns out to be measuring something else (the bug this replaces).
"""

import sys

import bpy
import numpy as np

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
tip = ebones.new("tip")
tip.parent = root
tip.use_connect = True
tip.head = root.tail
tip.tail = (0, 2, 0)
bpy.ops.object.mode_set(mode="POSE")
bpy.context.view_layer.update()

target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (1.0, 2.0, 0.0)  # off to the side of "tip" -- both root and tip are free to move
bpy.context.view_layer.update()


class Chain:
    tip_bone = "tip"
    chain_length = 0
    num_steps = 1000
    learning_rate = 0.2
    threshold = 0.0005
    patience = 200


class Item:
    obj_type = "LOOK_AT"
    enabled = True
    weight = 1.0
    target_object = target
    use_head = True
    bone_name = "tip"


chain = Chain()
chain.objectives = [Item()]

steps, obj, msg = bridge.solve_chain(armature_obj, chain)
print("steps=", steps, "obj=", obj)

bpy.context.view_layer.update()
pb = armature_obj.pose.bones["tip"]
head = np.array(pb.head)
tail = np.array(pb.tail)
bone_dir = tail - head
bone_dir = bone_dir / np.linalg.norm(bone_dir)
to_target = np.array(target.location) - head
to_target = to_target / np.linalg.norm(to_target)
angle = np.degrees(np.arccos(np.clip(np.dot(bone_dir, to_target), -1, 1)))
print("bone_dir=", bone_dir, "to_target=", to_target, "angle_deg=", angle)

# `obj` is the *total* combined loss (Look At + the hidden auto-regularizers,
# weight 0.05 each), so it now carries a nonzero floor from those even at a
# geometrically excellent solution -- the geometric check right below is the
# one that actually matters for "did Look At work".
check("optimizer converged", obj is not None and obj < 0.01, f"obj={obj}")
check("bone actually points at target (post-solve geometry)", angle < 3.0, f"angle_deg={angle:.2f}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
