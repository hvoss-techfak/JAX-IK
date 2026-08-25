import importlib

from . import bridge, handlers, operators, preferences, properties, ui

# Reloaded (not just imported) on every register(): reinstalling this
# extension over an already-running Blender session (e.g. `blender
# --command extension install-file ...`, or Install from Disk on top of an
# existing install) only replaces files on disk -- it does not touch this
# session's already-imported Python modules, which Python caches by name in
# sys.modules regardless of what's on disk. Toggling the add-on off/on then
# calls register() again, but on the *same, stale* module objects unless we
# force a re-read from disk here, so code changes would silently never take
# effect without a full Blender restart. importlib.reload() mutates each
# module object in place, so every other module's `from . import X`
# reference to it (bridge, operators, ui, ... all reference each other this
# way) sees the reloaded contents too, regardless of reload order.
_SUBMODULES = (properties, bridge, operators, preferences, ui, handlers)


def register():
    for module in _SUBMODULES:
        importlib.reload(module)

    properties.register()
    operators.register()
    preferences.register()
    ui.register()
    handlers.register()


def unregister():
    handlers.unregister()
    ui.unregister()
    preferences.unregister()
    operators.unregister()
    properties.unregister()
    bridge.clear_fk_solver_cache()
