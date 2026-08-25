#!/usr/bin/env bash
# (Re)populates wheels/ and blender_manifest.toml's `wheels` list for the
# JAX-IK Blender extension. Re-run this whenever jax-ik's pinned version in
# pyproject.toml changes, or Blender's supported platform/Python matrix
# changes -- nothing else in the add-on depends on how this step is run.
#
# Uses `peeler` (https://pypi.org/project/peeler/), a build-time-only tool
# (not a runtime dependency of the add-on), to resolve jax-ik's dependency
# tree with `uv` and download matching wheels.
set -euo pipefail
cd "$(dirname "$0")/.."  # blender_addon/jax_ik_blender/

if [ ! -d .peeler_venv ]; then
    python3 -m venv .peeler_venv
fi
# shellcheck disable=SC1091
source .peeler_venv/bin/activate
pip install --quiet --upgrade pip peeler uv

rm -rf wheels uv.lock
mkdir -p wheels

# jax-ik's own pyproject.toml declares mesh_to_sdf and pyvista as hard
# (non-extra) dependencies, but this add-on never imports either of them at
# runtime (see pyproject.toml's [tool.uv] override-dependencies comment for
# why that otherwise breaks wheel-only resolution entirely). Everything
# below EXCLUDED_PACKAGES is either one of those two, one of their own
# transitive dependencies, or (numpy/packaging) already bundled by Blender's
# own Python -- confirmed by tracing `import jax_ik.ik`'s actual sys.modules
# diff in a real interpreter, not just guessed from declared metadata.
EXCLUDED_PACKAGES=(
    numpy packaging
    attrs certifi colorama cycler cyclopts deprecated docstring-parser
    fonttools freetype-py idna imageio markdown-it-py mdurl mesh-to-sdf
    platformdirs pooch pycollada pyglet pyopengl pyparsing pyribbit
    python-dateutil pyvista requests rich rich-rst scooby six urllib3 wrapt
)

exclude_args=()
for pkg in "${EXCLUDED_PACKAGES[@]}"; do
    exclude_args+=(--exclude-package "$pkg")
done

peeler wheels ./pyproject.toml ./blender_manifest.toml ./wheels "${exclude_args[@]}"

deactivate
echo "Done. wheels/ populated and blender_manifest.toml's wheels list updated."
echo "Next: blender --command extension validate . && blender --command extension build --split-platforms"
