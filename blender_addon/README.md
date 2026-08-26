# JAX-IK for Blender

Poses armatures with [JAX-IK](https://github.com/hvoss-techfak/JAX-IK), a
differentiable, JAX-based inverse kinematics solver, from inside Blender.
`jax-ik` and its dependencies ship pre-bundled as part of the extension --
there is nothing to install separately.

## Install

1. Download/build `jax_ik_blender-<version>-linux_x64.zip` (see "Building the
   extension" below if you need to produce it yourself).
2. In Blender: Edit > Preferences > Get Extensions > the dropdown/gear menu
   in the top-right > "Install from Disk...", pick the zip, enable it.
3. That's it -- no dependency-install step. If something looks wrong, check
   Edit > Preferences > Add-ons > JAX-IK for a bundled-package version
   readout.

## Using it

1. Select an armature. Open the 3D viewport sidebar (press N) and click the
   **JAX-IK** tab.
2. Click **New JAX-IK Chain**, then set its **Tip Bone** (e.g. a hand or foot
   bone) in the field that appears -- this works whether or not you're in
   Pose Mode; if a pose bone is already selected/active it's used as the
   default.
3. Set **Chain Length** (how many bones up the hierarchy to control; 0 = all
   the way to the root, same convention as Blender's native IK).
4. Add an objective (the "+" under Objectives): pick a type, and for
   target-based types, an Empty (or any object) as the target. Add more than
   one to combine, e.g. "Reach Target" on a hand plus "Pole Target" to also
   control elbow direction.
5. Click **Solve**. Toggle **Live** to re-solve automatically while you drag
   the target object around -- off by default since every re-solve runs the
   full optimizer, which is slower than Blender's native IK on complex
   chains.
6. A status line appears below Solve/Live after every attempt (with a ⚠ icon
   on failure) explaining exactly what happened -- e.g. "no Target object
   set" or "Tip Bone is not set" -- instead of failing silently.

This panel lists every chain on the selected armature, so you can add,
inspect, and solve all of them in one place. Every enabled objective is
treated as mandatory (there's no soft/optional toggle) -- if you want a soft
pull instead, just give it a low **Weight**.

### Objective types

| Type | What it does |
|---|---|
| Reach Target | Pulls the bone's head/tail toward the target object -- the usual "IK to a point" behavior. |
| Look At | Orients the bone's head-to-tail axis at the target object, without trying to move the tip onto it. |
| Pole Target | Controls which way the chain bends (e.g. elbow/knee direction) toward the target object -- the standard two-bone pole-vector constraint. Only meaningful for chains of 3+ bones. |
| Avoid Point | Keeps the armature outside a sphere centered on the target object (radius set per-objective). Checks the space between each bone's joint and its parent's joint, across the *whole* armature (not just the controlled chain) -- so a chain's outermost bone's own far end (past its last joint) isn't covered unless a child bone continues past it. |
| Zero Rotation | Regularizer: pulls this chain's controlled joints toward their rest rotation. No target object. |
| Prefer Current Pose | Regularizer: minimizes this chain's movement away from the pose at solve start. No target object. |

Every solve also quietly adds Zero Rotation and Prefer Current Pose a
*second* time each, fixed at a low weight (0.05) and invisible in the
objectives list -- they're always on, on top of whatever you add explicitly
above. Neither is strong enough on its own to fight a real objective; they
just keep any axis a real objective doesn't pin down from drifting to an
arbitrary angle. Add either one explicitly (with its own **Weight**) if you
want a stronger pull than that baseline.

**"From Head" checkbox**: for Reach Target and Look At, this picks which end
of the bone the objective measures from -- checked uses the bone's head,
unchecked its tail.

### Bake to Keyframes

Each chain has its own **Start Frame** / **End Frame** fields (default
1-250; **Use Playback Range** copies the scene's playback range in), plus a
**Bake to Keyframes** button. Baking steps through that range frame by
frame -- End < Start bakes backward -- re-solving at each frame (so an
animated target is tracked correctly, since Blender's own depsgraph is
re-evaluated at every frame first) and inserting a `rotation_euler` keyframe
on every controlled bone. Each frame's solve starts from wherever the
previous baked frame left the pose, which combined with the always-on
"prefer current pose" regularizer (see below) gives some natural
frame-to-frame continuity rather than each frame solving from scratch. The
scene's original current frame is restored when baking finishes. If any
frame fails to solve, the bake stops there (frames already baked keep their
keyframes) and the status line says which frame and why.

## What "adheres to existing constraints" means here

- **Joint limits**: bounds come straight from Blender's own per-axis IK
  limit fields (`Lock`/`Limit` under a pose bone's IK panel) -- the same
  fields Blender's native IK constraint reads. Set those up as usual and
  JAX-IK will respect them.
- **Other constraints on the same bones**: JAX-IK only ever writes to a
  controlled bone's base rotation channel (`rotation_euler`), the same
  channel keyframes or manual posing would use. Any constraint stacked on
  that bone (Limit Rotation, Copy Rotation, ...) still runs normally
  afterward, through Blender's own evaluation.
- **Everything outside the controlled chain** (other chains, parents above
  the chain root, etc.): before each solve, JAX-IK reads where Blender's
  depsgraph has *currently* evaluated every other bone (i.e. after whatever
  constraints/drivers/other IK already did to it) and freezes that as fixed
  context for the solve. This is a per-solve snapshot, not a continuous,
  fully differentiable simulation of Blender's whole constraint stack --
  if something else moves the rig, click Solve again (or use Live) to pick
  up the new context.

## Limitations (v1)

- Each frame is solved independently (in sequence, seeded from the previous
  frame) -- no multi-frame trajectory objectives (derivative smoothing,
  equal-spacing) that optimize a whole clip jointly.
- No full-rotation "match this exact orientation, including roll" objective,
  and no mesh-derived self-collision (SDF) objective -- the latter needs a
  signed distance field rebuilt on every edit, which are expensive, so out of scope for a live posing tool. Avoid Point is a
  much cheaper, manually-placed-sphere substitute for simple obstacle
  avoidance, not full mesh self-collision.
- Only Linux x86_64 wheels are built/tested currently. See "Building the
  extension" to add other platforms -- no code changes needed, just
  re-running the wheel-fetch step for that platform.

## Building the extension

### Day to day: after editing the source

From `blender_addon/`:

```bash
./build_and_install.sh          # validate, build, install into your local Blender
./build_and_install.sh --test   # ...then also run the tests/*.py suite
```

Auto-detects a `blender` on `PATH` or a flatpak `org.blender.Blender`
install; override with `BLENDER=/path/to/blender` if it picks the wrong one.
If Blender is already open, disable and re-enable the add-on (Edit >
Preferences > Add-ons) afterward -- reinstalling only replaces files on
disk, it doesn't reload an already-running session's Python (this add-on's
`register()` does force a fresh reload of its own code, so a plain
disable/enable is enough; a full restart is never required).

### Only when dependencies change: rebuilding wheels

Dependencies are pre-bundled as wheels declared in `blender_manifest.toml`,
fetched with [`peeler`](https://pypi.org/project/peeler/) (a build-time-only
tool; it's not a runtime dependency of the add-on).

```bash
cd blender_addon/jax_ik_blender
./scripts/build_wheels.sh        # (re)populates wheels/ and blender_manifest.toml
```

Run this whenever `jax-ik`'s pinned version in `pyproject.toml` changes, or
to add wheels for another platform/Python version -- `build_and_install.sh`
above does not touch wheels, it just re-packages whatever is currently in
`wheels/` together with the current `.py` source.

## License

The blender addon for Jax-IK - 2025 by Hendric Voss is licensed under CC BY-NC-SA 4.0.
