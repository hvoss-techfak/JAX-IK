"""Verifies Avoid Point actually pushes the bone segment away from the
sphere obstacle by at least the requested radius (plus jax_ik's own small
built-in clearance/segment-radius padding)."""

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
# jax_ik's SphereCollisionPenaltyObjTraj checks segments between a bone's
# head and its *parent's* head -- i.e. it needs a child bone to "close" the
# measurement of the segment before it. A single parentless bone has no
# such segment at all, so this needs at least a 2-bone chain to test.
child = ebones.new("child")
child.parent = root
child.use_connect = True
child.head = root.tail
child.tail = (0, 2, 0)
bpy.ops.object.mode_set(mode="POSE")
bpy.context.view_layer.update()

obstacle = bpy.data.objects.new("Obstacle", None)
bpy.context.collection.objects.link(obstacle)
obstacle.location = (0, 0.5, 0.1)  # 0.1 from the bone's straight-line path
radius = 0.3
bpy.context.view_layer.update()


class Chain:
    tip_bone = "child"
    chain_length = 0
    num_steps = 1000
    learning_rate = 0.2
    threshold = 0.0005
    patience = 200


class Item:
    obj_type = "AVOID_SPHERE"
    enabled = True
    weight = 1.0
    avoid_radius = radius

    def __init__(self, t):
        self.target_object = t


chain = Chain()
chain.objectives = [Item(obstacle)]

steps, obj, msg = bridge.solve_chain(armature_obj, chain)
bpy.context.view_layer.update()

pb = armature_obj.pose.bones["root"]
head = np.array(pb.head)
tail = np.array(pb.tail)
center = np.array(obstacle.location)

# Closest distance from the obstacle center to the bone segment.
seg = tail - head
t = np.clip(np.dot(center - head, seg) / np.dot(seg, seg), 0.0, 1.0)
closest = head + t * seg
dist = float(np.linalg.norm(center - closest))

print(f"steps={steps} obj={obj} head={head} tail={tail} closest_dist={dist:.4f} (radius={radius})")
check("solve returned a result", obj is not None)
check("bone segment cleared the obstacle radius", dist >= radius - 1e-3, f"dist={dist:.4f} radius={radius}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
