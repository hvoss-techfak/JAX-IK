"""Verifies solve_chain's new diagnostic messages -- in particular the case
this was added for: an objective whose obj_type is a value that no longer
exists in the current OBJECTIVE_TYPES enum (e.g. "DIRECTION"/"ORIENTATION",
removed in this update). This is the most likely explanation for "the Solve
button does nothing, and no error is visible": Blender enums are stored
positionally, so a chain/objective created before this update can silently
end up pointing at a removed or renumbered type after the add-on's Python
module is updated in an already-open session, and previously this failed
completely silently (a status-bar-only warning, easy to miss, and Live mode
only printed to the terminal).
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
bpy.context.view_layer.update()


class Chain:
    tip_bone = "root"
    chain_length = 0
    num_steps = 100
    learning_rate = 0.2
    threshold = 0.0005
    patience = 100


class StaleItem:
    obj_type = "DIRECTION"  # removed in this update -- simulates leftover data
    enabled = True
    weight = 1.0
    target_object = None
    use_head = True
    bone_name = ""


class NoTipChain(Chain):
    tip_bone = ""


class EmptyChain(Chain):
    pass


# Case 1: unrecognized (stale) objective type.
chain1 = Chain()
chain1.objectives = [StaleItem()]
steps, obj, msg = bridge.solve_chain(armature_obj, chain1)
print("case1:", steps, obj, msg)
check("stale type -> nothing solved", obj is None)
check("stale type -> message names it", "unrecognized type" in msg and "DIRECTION" in msg, msg)

# Case 2: no objectives at all.
chain2 = EmptyChain()
chain2.objectives = []
steps, obj, msg = bridge.solve_chain(armature_obj, chain2)
print("case2:", steps, obj, msg)
check("no objectives -> nothing solved", obj is None)
check("no objectives -> clear message", msg == "No objectives added yet.", msg)

# Case 3: tip bone not set.
chain3 = NoTipChain()
chain3.objectives = []
steps, obj, msg = bridge.solve_chain(armature_obj, chain3)
print("case3:", steps, obj, msg)
check("no tip bone -> nothing solved", obj is None)
check("no tip bone -> clear message", "Tip Bone" in msg, msg)

# Case 4: a valid, enabled objective with no target object set.
class NoTargetItem:
    obj_type = "DISTANCE"
    enabled = True
    weight = 1.0
    target_object = None
    use_head = True
    bone_name = ""


chain4 = Chain()
chain4.objectives = [NoTargetItem()]
steps, obj, msg = bridge.solve_chain(armature_obj, chain4)
print("case4:", steps, obj, msg)
check("no target -> nothing solved", obj is None)
check("no target -> message names it", "no Target object set" in msg, msg)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
