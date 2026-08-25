"""Headless smoke test for the JAX-IK Blender extension.

Run with:
    blender --background --python blender_addon/tests/smoke_test.py

Builds a tiny 3-bone armature, adds a JAX-IK chain with a Reach Target
objective, solves, and asserts the tip bone actually reaches the target.
Also exercises IK-limit bounds and the "freeze uncontrolled bones" behavior.
"""

import math
import sys

import bpy
import mathutils

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def main():
    addon_mod = "bl_ext.user_default.jax_ik_blender"
    if addon_mod not in sys.modules:
        try:
            import addon_utils

            result = addon_utils.enable(addon_mod, default_set=True, persistent=True)
            check("addon enabled via addon_utils", result is not None, repr(result))
        except Exception as exc:  # noqa: BLE001
            check("addon enabled via addon_utils", False, str(exc))
            print(FAILURES)
            sys.exit(1)

    from bl_ext.user_default.jax_ik_blender import bridge

    available, err = bridge.is_available()
    check("jax_ik importable", available, err)
    if not available:
        print("Cannot continue without jax_ik.")
        sys.exit(1)

    # --- build a tiny 3-bone chain: root -> mid -> tip -------------------
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.name = "TestArmature"
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

    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0.5, 2.2, 0.3)
    bpy.context.view_layer.update()

    chains = armature_obj.data.jax_ik_chains
    chain = chains.add()
    chain.tip_bone = "tip"
    chain.chain_length = 3  # whole chain (root, mid, tip) -- chain_length has a hard min of 1 now
    chain.num_steps = 2000
    chain.threshold = 0.0005
    chain.patience = 300

    item = chain.objectives.add()
    item.obj_type = "DISTANCE"
    item.target_object = target
    item.use_head = False  # use tip's tail = the very end of the chain
    item.weight = 1.0

    steps, obj_value, msg = bridge.solve_chain(armature_obj, chain)
    check("solve_chain returned a result", obj_value is not None, f"steps={steps} obj={obj_value}")

    bpy.context.view_layer.update()
    tip_pose = armature_obj.pose.bones["tip"]
    tip_tail_world = armature_obj.matrix_world @ tip_pose.tail
    dist = (tip_tail_world - target.matrix_world.translation).length
    check("tip reaches target", dist < 0.05, f"dist={dist:.4f}")

    # --- IK limit bounds are respected ------------------------------------
    mid_pb = armature_obj.pose.bones["mid"]
    mid_pb.use_ik_limit_x = True
    mid_pb.ik_min_x = 0.0
    mid_pb.ik_max_x = 0.0
    bridge.clear_fk_solver_cache()
    steps2, obj_value2, msg2 = bridge.solve_chain(armature_obj, chain)
    check("solve with locked axis returned a result", obj_value2 is not None)
    check(
        "locked axis stayed at 0",
        abs(mid_pb.rotation_euler[0]) < 1e-5,
        f"mid.rotation_euler.x={mid_pb.rotation_euler[0]}",
    )

    # --- unrelated bone's manual pose is preserved after solving ---------
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.armature_add(enter_editmode=True, location=(3, 0, 0))
    other_arm = bpy.context.object
    other_arm.name = "OtherArmature"
    other_ebones = other_arm.data.edit_bones
    ob_root = other_ebones[0]
    ob_root.name = "root"
    ob_child = other_ebones.new("child")
    ob_child.parent = ob_root
    ob_child.use_connect = True
    ob_child.head = ob_root.tail
    ob_child.tail = ob_root.tail + mathutils.Vector((0, 1, 0))
    bpy.ops.object.mode_set(mode="POSE")

    ob_root_pb = other_arm.pose.bones["root"]
    ob_root_pb.rotation_mode = "XYZ"
    ob_root_pb.rotation_euler = (0.3, 0.0, 0.0)
    bpy.context.view_layer.update()

    other_target = bpy.data.objects.new("OtherTarget", None)
    bpy.context.collection.objects.link(other_target)
    other_target.location = other_arm.pose.bones["child"].tail + mathutils.Vector((0.1, 0.1, 0.0))
    bpy.context.view_layer.update()

    other_chains = other_arm.data.jax_ik_chains
    other_chain = other_chains.add()
    other_chain.tip_bone = "child"
    other_chain.chain_length = 1  # only "child" is controlled, "root" stays frozen
    other_chain.num_steps = 300
    other_item = other_chain.objectives.add()
    other_item.obj_type = "DISTANCE"
    other_item.target_object = other_target
    other_item.use_head = False

    before = tuple(ob_root_pb.rotation_euler)
    bridge.solve_chain(other_arm, other_chain)
    after = tuple(ob_root_pb.rotation_euler)
    check(
        "uncontrolled bone's manual pose preserved",
        all(abs(a - b) < 1e-5 for a, b in zip(before, after)),
        f"before={before} after={after}",
    )

    print("\n=== SUMMARY ===")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


main()
