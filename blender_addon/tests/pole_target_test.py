"""Verifies Pole Target actually controls bend direction: solving the same
under-constrained 3-bone chain (Reach Target alone leaves the bend direction
ambiguous) with the pole placed on the +X side should bend the elbow toward
+X, and with the pole on -X should bend it toward -X.
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


def build_armature():
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.mode != "OBJECT" else None
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature_obj = bpy.context.object
    ebones = armature_obj.data.edit_bones
    root = ebones[0]
    root.name = "root"
    root.head = (0, 0, 0)
    root.tail = (0, 1, 0)
    mid = ebones.new("mid")
    mid.parent = root
    mid.use_connect = True
    mid.head = root.tail
    mid.tail = (0, 2, 0)
    tip = ebones.new("tip")
    tip.parent = mid
    tip.use_connect = True
    tip.head = mid.tail
    tip.tail = (0, 3, 0)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.context.view_layer.update()
    return armature_obj


class Chain:
    tip_bone = "tip"
    chain_length = 0
    num_steps = 1500
    learning_rate = 0.2
    threshold = 0.0005
    patience = 300


class ReachItem:
    obj_type = "DISTANCE"
    enabled = True
    weight = 1.0
    use_head = True
    bone_name = "tip"

    def __init__(self, target):
        self.target_object = target


class PoleItem:
    obj_type = "POLE_TARGET"
    enabled = True
    weight = 1.0
    bone_name = ""

    def __init__(self, target):
        self.target_object = target


def run(pole_x):
    armature_obj = build_armature()

    reach_target = bpy.data.objects.new("ReachTarget", None)
    bpy.context.collection.objects.link(reach_target)
    # Deliberately off the dead-straight rest-pose axis: an exactly-on-axis
    # target closer than full extension is a classic gradient-descent
    # local-minimum trap for *any* IK objective (the "fold to get closer"
    # direction has ~zero initial gradient from a perfectly straight start)
    # -- unrelated to Pole Target, so avoid it here to isolate what this
    # test actually checks.
    reach_target.location = (0.3, 1.6, 0)

    pole_target = bpy.data.objects.new("PoleTarget", None)
    bpy.context.collection.objects.link(pole_target)
    pole_target.location = (pole_x, 1.0, 0)
    bpy.context.view_layer.update()

    chain = Chain()
    chain.objectives = [ReachItem(reach_target), PoleItem(pole_target)]

    bridge.clear_fk_solver_cache()
    steps, obj, msg = bridge.solve_chain(armature_obj, chain)
    bpy.context.view_layer.update()
    mid_head = np.array(armature_obj.pose.bones["mid"].head)
    tip_head = np.array(armature_obj.pose.bones["tip"].head)
    print(f"pole_x={pole_x} steps={steps} obj={obj} mid_head={mid_head} tip_head={tip_head}")
    return mid_head, obj


mid_pos_pole, obj_pos = run(pole_x=1.0)
mid_neg_pole, obj_neg = run(pole_x=-1.0)

# Reach Target and Pole Target are competing objectives here (reach exactly
# vs. bend a specific way), so this isn't checking tight convergence -- just
# that the optimizer made real progress rather than getting stuck.
check("solve with +X pole made progress", obj_pos is not None and obj_pos < 0.1, f"obj={obj_pos}")
check("solve with -X pole made progress", obj_neg is not None and obj_neg < 0.1, f"obj={obj_neg}")
check("+X pole bends elbow toward +X", mid_pos_pole[0] > 0.05, f"mid.head.x={mid_pos_pole[0]:.4f}")
check("-X pole bends elbow toward -X", mid_neg_pole[0] < -0.05, f"mid.head.x={mid_neg_pole[0]:.4f}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
