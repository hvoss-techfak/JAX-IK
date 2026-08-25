"""Verifies Bake to Keyframes: solves a chain across a frame range against an
*animated* target (so per-frame re-evaluation actually matters) and checks
keyframes landed on every frame with the tip tracking the target at both
ends of the range. Also checks backward baking (End < Start).
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
bpy.context.view_layer.objects.active = armature_obj
bpy.context.view_layer.update()

target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (0.3, 1.6, 0.2)
target.keyframe_insert(data_path="location", frame=1)
target.location = (-0.3, 1.5, 0.3)
target.keyframe_insert(data_path="location", frame=10)
bpy.context.view_layer.update()

r = bpy.ops.jaxik.add_chain()
check("add_chain", r == {"FINISHED"})
armature_obj.data.jax_ik_chains[-1].tip_bone = "tip"
chain_index = len(armature_obj.data.jax_ik_chains) - 1
chain = armature_obj.data.jax_ik_chains[chain_index]
chain.chain_length = 2  # whole chain (root, tip) -- chain_length has a hard min of 1 now
chain.bake_start_frame = 1
chain.bake_end_frame = 10
chain.num_steps = 500
chain.threshold = 0.0005

bpy.ops.jaxik.add_objective(chain_index=chain_index)
item = chain.objectives[-1]
item.obj_type = "DISTANCE"
item.target_object = target
item.use_head = False

r = bpy.ops.jaxik.bake(chain_index=chain_index)
check("bake op finished", r == {"FINISHED"}, str(r))
print("last_status:", chain.last_status)
check("last_status reports 10 frames", "Baked 10 frame" in chain.last_status, chain.last_status)

def _iter_fcurves(action):
    # Blender 4.4+ layered/slotted Action model: fcurves live under
    # layers -> strips -> channelbags, not directly on the Action.
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fc in channelbag.fcurves:
                    yield fc


def _tip_rotation_keyframe_frames(action):
    frames = set()
    for fc in _iter_fcurves(action):
        if fc.data_path == 'pose.bones["tip"].rotation_euler':
            for kp in fc.keyframe_points:
                frames.add(round(kp.co[0]))
    return frames


action = armature_obj.animation_data.action if armature_obj.animation_data else None
check("armature has an action after baking", action is not None)
if action is not None:
    fcurve_frames = _tip_rotation_keyframe_frames(action)
    check("keyframes exist on every frame 1..10", fcurve_frames == set(range(1, 11)), sorted(fcurve_frames))


def tip_world_at(frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    return np.array(armature_obj.matrix_world @ armature_obj.pose.bones["tip"].tail)


d1 = np.linalg.norm(tip_world_at(1) - np.array((0.3, 1.6, 0.2)))
d10 = np.linalg.norm(tip_world_at(10) - np.array((-0.3, 1.5, 0.3)))
print("dist at frame1:", d1, "dist at frame10:", d10)
# Loosened from a tighter tolerance now that the auto-added Zero
# Rotation/Prefer Current Pose Optional objectives pull harder (weight 0.05,
# up from 0.01) -- trading a bit of precision for stability is the whole
# point of that change, so this just confirms "closely tracks", not
# "converges to the old, tighter precision".
check("tip tracks animated target at frame 1", d1 < 0.1, f"d1={d1:.4f}")
check("tip tracks animated target at frame 10", d10 < 0.1, f"d10={d10:.4f}")

# --- backward bake: should still produce the same frame coverage ----------
for fc in list(_iter_fcurves(action)):
    if fc.data_path == 'pose.bones["tip"].rotation_euler':
        for channelbag in (cb for layer in action.layers for strip in layer.strips for cb in strip.channelbags):
            if fc in list(channelbag.fcurves):
                channelbag.fcurves.remove(fc)
                break

chain.bake_start_frame = 10
chain.bake_end_frame = 1
r = bpy.ops.jaxik.bake(chain_index=chain_index)
check("backward bake finished", r == {"FINISHED"}, str(r))
print("last_status (backward):", chain.last_status)
check("backward status reports 10 frames", "Baked 10 frame" in chain.last_status, chain.last_status)

fcurve_frames = _tip_rotation_keyframe_frames(action)
check("backward bake keyframed every frame 1..10", fcurve_frames == set(range(1, 11)), sorted(fcurve_frames))

# --- autosmooth: keyframes every frame, reports smoothed, still tracks ----
for fc in list(_iter_fcurves(action)):
    if fc.data_path == 'pose.bones["tip"].rotation_euler':
        for channelbag in (cb for layer in action.layers for strip in layer.strips for cb in strip.channelbags):
            if fc in list(channelbag.fcurves):
                channelbag.fcurves.remove(fc)
                break

chain.bake_start_frame = 1
chain.bake_end_frame = 10
chain.bake_autosmooth = True
chain.bake_autosmooth_amount = 0.6
r = bpy.ops.jaxik.bake(chain_index=chain_index)
check("autosmooth bake finished", r == {"FINISHED"}, str(r))
print("last_status (autosmooth):", chain.last_status)
check("autosmooth status reports 10 frames", "Baked 10 frame" in chain.last_status, chain.last_status)
check("autosmooth status mentions smoothing", "smoothed" in chain.last_status, chain.last_status)

fcurve_frames = _tip_rotation_keyframe_frames(action)
check("autosmooth bake keyframed every frame 1..10", fcurve_frames == set(range(1, 11)), sorted(fcurve_frames))

d1_smooth = np.linalg.norm(tip_world_at(1) - np.array((0.3, 1.6, 0.2)))
d10_smooth = np.linalg.norm(tip_world_at(10) - np.array((-0.3, 1.5, 0.3)))
print("autosmooth dist at frame1:", d1_smooth, "dist at frame10:", d10_smooth)
# Looser than the unsmoothed checks above: smoothing deliberately trades a
# bit of per-frame precision for cross-frame continuity, so this only
# confirms the tip still broadly tracks, not that it matches the
# unsmoothed precision.
check("autosmooth tip still roughly tracks target at frame 1", d1_smooth < 0.2, f"d1={d1_smooth:.4f}")
check("autosmooth tip still roughly tracks target at frame 10", d10_smooth < 0.2, f"d10={d10_smooth:.4f}")

chain.bake_autosmooth = False  # leave the chain in its default state

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
