"""Regression test for the chain-length/tip-bone topology-change bone reset:
when a chain's controlled-bone set shrinks (chain_length reduced), bones
that drop out of control must be reset to their pre-chain rotation, not left
at whatever a previous solve last set them to -- otherwise that stale
rotation silently corrupts the FK baseline of the *next* solve. See
bridge.sync_chain_bone_snapshots / properties._on_chain_topology_changed.
"""

import math
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


# --- build a 3-bone chain: root -> mid -> tip ------------------------------
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
bpy.context.view_layer.objects.active = armature_obj
bpy.context.view_layer.update()

target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (0.6, 2.0, 0.4)
bpy.context.view_layer.update()

r = bpy.ops.jaxik.add_chain()
check("add_chain", r == {"FINISHED"}, str(r))
chain = armature_obj.data.jax_ik_chains[-1]
chain_index = len(armature_obj.data.jax_ik_chains) - 1
chain.tip_bone = "tip"
chain.chain_length = 3  # whole chain: root, mid, tip
chain.num_steps = 1500
chain.threshold = 0.0005
chain.patience = 300

check(
    "snapshot tracks all 3 controlled bones after growing to chain_length=3",
    {s.bone_name for s in chain.bone_snapshots} == {"root", "mid", "tip"},
    sorted(s.bone_name for s in chain.bone_snapshots),
)

r = bpy.ops.jaxik.add_objective(chain_index=chain_index)
check("add_objective", r == {"FINISHED"}, str(r))
item = chain.objectives[-1]
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

steps, obj_value, msg = bridge.solve_chain(armature_obj, chain)
check("solve returned a result", obj_value is not None, f"steps={steps} obj={obj_value} msg={msg}")

mid_pb = armature_obj.pose.bones["mid"]
root_pb = armature_obj.pose.bones["root"]
mid_after_solve = tuple(mid_pb.rotation_euler)
root_after_solve = tuple(root_pb.rotation_euler)
print("mid rotation after solve:", mid_after_solve)
print("root rotation after solve:", root_after_solve)
check(
    "solve actually moved mid/root away from rest",
    any(abs(a) > 1e-3 for a in mid_after_solve) or any(abs(a) > 1e-3 for a in root_after_solve),
    f"mid={mid_after_solve} root={root_after_solve}",
)

# --- shrink chain_length: mid/root drop out of control, must reset --------
chain.chain_length = 1  # now only "tip" is controlled

check(
    "snapshot only tracks the tip bone after shrinking to chain_length=1",
    {s.bone_name for s in chain.bone_snapshots} == {"tip"},
    sorted(s.bone_name for s in chain.bone_snapshots),
)

mid_after_shrink = tuple(mid_pb.rotation_euler)
root_after_shrink = tuple(root_pb.rotation_euler)
print("mid rotation after shrink:", mid_after_shrink)
print("root rotation after shrink:", root_after_shrink)
check(
    "mid was reset to its pre-chain (rest) rotation after dropping out of control",
    all(abs(a) < 1e-5 for a in mid_after_shrink),
    f"mid_after_shrink={mid_after_shrink}",
)
check(
    "root was reset to its pre-chain (rest) rotation after dropping out of control",
    all(abs(a) < 1e-5 for a in root_after_shrink),
    f"root_after_shrink={root_after_shrink}",
)

# --- next solve's FK baseline must actually reflect the reset, not the ------
# --- stale solved rotation (the whole point of the reset) ------------------
steps2, obj_value2, msg2 = bridge.solve_chain(armature_obj, chain)
check("second solve (tip-only) returned a result", obj_value2 is not None, f"steps={steps2} obj={obj_value2}")
mid_after_second_solve = tuple(mid_pb.rotation_euler)
check(
    "mid (now uncontrolled) stayed frozen at its reset rotation through the next solve",
    all(abs(a) < 1e-5 for a in mid_after_second_solve),
    f"mid_after_second_solve={mid_after_second_solve}",
)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
