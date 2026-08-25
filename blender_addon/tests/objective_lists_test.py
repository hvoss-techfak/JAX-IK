"""Regression test for the single-objective-list + Optional-toggle design:
jaxik.add_objective / jaxik.remove_objective / jaxik.move_objective all work
on chain.objectives, and each row's `optional` bool (not a separate
CollectionProperty) decides whether it's Mandatory or Optional for the
solve. Also checks that a new chain's objectives list starts empty -- the
Zero Rotation / Prefer Current Pose stabilizers are hardcoded, invisible,
and always-on inside bridge.solve_chain itself, not stored as rows here.
"""

import sys

import bpy

import addon_utils

addon_utils.enable("bl_ext.user_default.jax_ik_blender", default_set=True, persistent=True)

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

r = bpy.ops.jaxik.add_chain()
check("add_chain", r == {"FINISHED"}, str(r))
chain = armature_obj.data.jax_ik_chains[-1]
chain_index = len(armature_obj.data.jax_ik_chains) - 1

check(
    "new chain's objectives list starts empty (auto-regularizers are hidden, not rows)",
    len(chain.objectives) == 0,
    len(chain.objectives),
)

# --- add_objective appends to the list, defaulting to mandatory ------------
r = bpy.ops.jaxik.add_objective(chain_index=chain_index)
check("add objective", r == {"FINISHED"}, str(r))
check("list grew to 1", len(chain.objectives) == 1, len(chain.objectives))
check("new objective defaults to mandatory (optional=False)", chain.objectives[-1].optional is False, chain.objectives[-1].optional)
check("active index followed the add", chain.active_objective_index == 0, chain.active_objective_index)

chain.objectives[-1].obj_type = "ZERO_ROTATION"
bpy.ops.jaxik.add_objective(chain_index=chain_index)
chain.objectives[-1].obj_type = "PREFER_CURRENT"
bpy.ops.jaxik.add_objective(chain_index=chain_index)
chain.objectives[-1].obj_type = "AVOID_SPHERE"
check("list grew to 3", len(chain.objectives) == 3, len(chain.objectives))

# --- the `optional` flag is a plain, freely-toggleable bool -----------------
chain.objectives[-1].optional = True
check("optional flag can be toggled on", chain.objectives[-1].optional is True, chain.objectives[-1].optional)
chain.objectives[-1].optional = False
check("optional flag can be toggled back off", chain.objectives[-1].optional is False, chain.objectives[-1].optional)

# --- move_objective reorders within the list --------------------------------
before_order = [o.obj_type for o in chain.objectives]
r = bpy.ops.jaxik.move_objective(chain_index=chain_index, direction="UP")
check("move objective up", r == {"FINISHED"}, str(r))
after_order = [o.obj_type for o in chain.objectives]
check(
    "move swapped the last two entries",
    after_order == [before_order[0], before_order[2], before_order[1]],
    f"before={before_order} after={after_order}",
)
check("active index followed the move", chain.active_objective_index == 1, chain.active_objective_index)

# Moving the first item further up is a no-op (already at the top).
chain.active_objective_index = 0
r = bpy.ops.jaxik.move_objective(chain_index=chain_index, direction="UP")
check("move first item up is a no-op cancel", r == {"CANCELLED"}, str(r))

# --- remove_objective removes from the list ---------------------------------
before_len = len(chain.objectives)
r = bpy.ops.jaxik.remove_objective(chain_index=chain_index, objective_index=0)
check("remove objective", r == {"FINISHED"}, str(r))
check("list shrank", len(chain.objectives) == before_len - 1, len(chain.objectives))

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
