"""Live-update support: re-solve chains with `live_update` enabled whenever
the scene changes (e.g. the user drags a target Empty).

Kept deliberately simple (re-check every armature on every depsgraph update,
rather than diffing exactly what changed) since Live is opt-in and off by
default per chain -- see JaxIKChain.live_update.
"""

import bpy

from . import bridge

_is_solving = False


def _on_depsgraph_update_post(scene, depsgraph):
    global _is_solving
    if _is_solving:
        return

    _is_solving = True
    try:
        for obj in bpy.data.objects:
            if obj.type != "ARMATURE":
                continue
            chains = getattr(obj.data, "jax_ik_chains", None)
            if not chains:
                continue
            for chain in chains:
                if not (chain.enabled and chain.live_update and chain.tip_bone):
                    continue
                try:
                    steps, obj_value, message = bridge.solve_chain(obj, chain)
                except Exception as exc:  # noqa: BLE001 - never let a live-solve crash Blender
                    chain.last_status = f"Live solve failed: {exc}"
                    chain.last_status_is_error = True
                    print(f"JAX-IK live solve failed for chain '{chain.name}': {exc}")
                    continue
                if obj_value is None:
                    chain.last_status = f"Live: nothing to solve: {message}"
                    chain.last_status_is_error = True
                else:
                    chain.last_status = f"Live: solved in {steps} steps (objective={obj_value:.5f})"
                    chain.last_status_is_error = False
    finally:
        _is_solving = False


def register():
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)


def unregister():
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
