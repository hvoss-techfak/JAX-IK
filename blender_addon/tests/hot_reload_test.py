"""Regression test for the actual root cause behind "it worked, then a
later change made it stop working, with no visible error": reinstalling
this extension (`install-file`, or Install from Disk) while Blender is
already running only replaces files on disk -- it does not touch that
session's already-imported Python modules, which Python caches by name
regardless of what's on disk. Toggling the add-on off/on re-runs
register()/unregister(), but on the *same* stale module objects unless
register() explicitly reloads them from disk.

This test mutates the *installed* copy of properties.py directly (with a
marker constant), simulating "the files on disk changed", then does a
real disable/enable cycle (exactly what a user should do after
reinstalling) and checks the running module actually picked up the change
-- proving __init__.py's importlib.reload() fix works, not just that it's
present in the source. The installed file is restored afterward regardless
of pass/fail.
"""

import shutil
import sys

import bpy

import addon_utils

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


MOD = "bl_ext.user_default.jax_ik_blender"

r = addon_utils.enable(MOD, default_set=True, persistent=True)
check("initial enable", r is not None)

from bl_ext.user_default.jax_ik_blender import properties

properties_file = properties.__file__
backup_path = properties_file + ".bak"
shutil.copy(properties_file, backup_path)

try:
    for marker in ("HOT_RELOAD_MARKER_V1", "HOT_RELOAD_MARKER_V2"):
        with open(properties_file, "a") as f:
            f.write(f"\n{marker} = True\n")

        # What a user should do after reinstalling: disable, then enable
        # again -- not a Blender restart, just toggling the add-on.
        addon_utils.disable(MOD, default_set=True)
        r = addon_utils.enable(MOD, default_set=True, persistent=True)
        check(f"re-enable after editing file ({marker})", r is not None)

        from bl_ext.user_default.jax_ik_blender import properties as reloaded_properties

        check(
            f"running module picked up {marker} after disable/enable",
            getattr(reloaded_properties, marker, False) is True,
            f"module file: {reloaded_properties.__file__}",
        )

    # A disable/enable cycle (now involving reload) must not itself break
    # class (re)registration -- e.g. "already registered" errors if old
    # classes from a previous register() weren't properly unregistered.
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.data.edit_bones[0].name = "root"
    bpy.ops.object.mode_set(mode="POSE")
    r = bpy.ops.jaxik.add_chain()
    check("operators still work after repeated reload cycles", r == {"FINISHED"}, str(r))

finally:
    # Restore the file, then cycle through a *real* disable/enable so the
    # reload happens through the normal register()/unregister() pairing
    # (unregister the currently-registered classes first, reload, then
    # register the freshly-reloaded ones) -- reloading directly here
    # instead would leave a generation of classes in the module that were
    # never actually registered, which is a self-inflicted test artifact,
    # not something a real disable/enable cycle would ever produce.
    shutil.move(backup_path, properties_file)
    addon_utils.disable(MOD, default_set=True)
    addon_utils.enable(MOD, default_set=True, persistent=True)

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
