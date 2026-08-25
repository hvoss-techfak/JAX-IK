"""Diagnostics-only preferences: jax-ik ships as bundled Extension wheels
(see blender_manifest.toml), so there is nothing to install here -- just a
place to confirm the bundled versions when troubleshooting.
"""

import importlib.metadata

import bpy

_REPORTED_PACKAGES = ("jax-ik", "jax", "jaxlib", "numpy", "trimesh", "urchin", "pygltflib")


class JAXIK_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        from . import bridge

        layout = self.layout
        available, error = bridge.is_available()

        if available:
            layout.label(text="jax-ik is available.", icon="CHECKMARK")
            col = layout.column()
            for name in _REPORTED_PACKAGES:
                try:
                    version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    version = "not found"
                col.label(text=f"{name}: {version}")
        else:
            layout.label(text="jax-ik could not be imported.", icon="ERROR")
            layout.label(text=error)
            layout.label(text="Re-installing the extension (Install from Disk) usually fixes this.")


classes = (JAXIK_AddonPreferences,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
