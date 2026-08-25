#!/usr/bin/env bash
# Rebuilds the JAX-IK Blender extension from the current source
# (jax_ik_blender/*.py + its already-fetched wheels/) and (re)installs it
# into your local Blender. Run this after any source edit; use it any time
# you want a fresh build+install without re-typing the manual
# validate/build/install-file sequence.
#
# This does NOT touch dependency wheels -- if jax-ik's pinned version in
# jax_ik_blender/pyproject.toml changes, or you need wheels for a different
# platform/Python version, run jax_ik_blender/scripts/build_wheels.sh first;
# this script just re-packages whatever is currently in wheels/.
#
# The one exception is jax-ik itself: --jax-ik local swaps the bundled
# jax-ik wheel for one built fresh from this repo's own src/jax_ik (e.g. to
# test unreleased changes in the add-on before they're published to PyPI),
# instead of the PyPI-published one build_wheels.sh normally fetches.
#
# Usage:
#   ./build_and_install.sh                    # build + install
#   ./build_and_install.sh --jax-ik local      # ...using local src/jax_ik instead of PyPI
#   ./build_and_install.sh --test              # build + install, then run tests/*.py
#   ./build_and_install.sh --no-split          # one zip for all platforms, not just this one
#
# Env vars:
#   BLENDER                      Path/command to run Blender (default: auto-detect)
#   JAX_IK_BLENDER_PLATFORM      Platform tag to pick out of a split build
#                                 (default: linux-x64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$SCRIPT_DIR/jax_ik_blender"
BUILD_DIR="$SCRIPT_DIR/build"
WHEELS_DIR="$SOURCE_DIR/wheels"
MANIFEST="$SOURCE_DIR/blender_manifest.toml"
EXTENSION_ID="jax_ik_blender"
REPO="user_default"
PLATFORM="${JAX_IK_BLENDER_PLATFORM:-linux-x64}"

RUN_TESTS=0
SPLIT_PLATFORMS=1
JAX_IK_SOURCE="pypi"
while [ $# -gt 0 ]; do
    case "$1" in
        --test) RUN_TESTS=1; shift ;;
        --no-split) SPLIT_PLATFORMS=0; shift ;;
        --jax-ik)
            [ $# -ge 2 ] || { echo "--jax-ik needs an argument: pypi or local" >&2; exit 1; }
            JAX_IK_SOURCE="$2"
            shift 2
            ;;
        --jax-ik=*)
            JAX_IK_SOURCE="${1#--jax-ik=}"
            shift
            ;;
        -h|--help)
            sed -n '2,27p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1 (see --help)" >&2
            exit 1
            ;;
    esac
done
case "$JAX_IK_SOURCE" in
    pypi|local) ;;
    *)
        echo "--jax-ik must be 'pypi' or 'local', got: $JAX_IK_SOURCE" >&2
        exit 1
        ;;
esac

if [ "$JAX_IK_SOURCE" = "local" ]; then
    echo "==> Building jax-ik from local source ($REPO_ROOT/src/jax_ik)"
    command -v uv >/dev/null 2>&1 || {
        echo "uv is required to build the local jax-ik wheel (https://docs.astral.sh/uv/)." >&2
        exit 1
    }
    OLD_WHEEL="$(ls "$WHEELS_DIR"/jax_ik-*.whl 2>/dev/null | head -n1 || true)"
    if [ -z "$OLD_WHEEL" ]; then
        echo "No existing jax_ik-*.whl in $WHEELS_DIR -- run" \
             "jax_ik_blender/scripts/build_wheels.sh first (it fetches every" \
             "*other* dependency's wheels too, not just jax-ik's)." >&2
        exit 1
    fi

    LOCAL_WHEEL_TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$LOCAL_WHEEL_TMPDIR"' EXIT
    (cd "$REPO_ROOT" && uv build --wheel --out-dir "$LOCAL_WHEEL_TMPDIR")
    NEW_WHEEL="$(ls "$LOCAL_WHEEL_TMPDIR"/jax_ik-*.whl | head -n1)"

    OLD_WHEEL_NAME="$(basename "$OLD_WHEEL")"
    NEW_WHEEL_NAME="$(basename "$NEW_WHEEL")"
    rm -f "$OLD_WHEEL"
    cp "$NEW_WHEEL" "$WHEELS_DIR/$NEW_WHEEL_NAME"
    if [ "$OLD_WHEEL_NAME" != "$NEW_WHEEL_NAME" ]; then
        sed -i "s|\./wheels/${OLD_WHEEL_NAME}|./wheels/${NEW_WHEEL_NAME}|" "$MANIFEST"
    fi
    echo "==> Using local wheel: $NEW_WHEEL_NAME (replaced $OLD_WHEEL_NAME)"
fi

# Locate a Blender to drive. Override with BLENDER=/path/to/blender if
# auto-detection picks the wrong one (e.g. you have more than one install).
if [ -n "${BLENDER:-}" ]; then
    read -r -a BLENDER_CMD <<< "$BLENDER"
elif command -v blender >/dev/null 2>&1; then
    BLENDER_CMD=(blender)
elif command -v flatpak >/dev/null 2>&1 && flatpak info org.blender.Blender >/dev/null 2>&1; then
    BLENDER_CMD=(flatpak run org.blender.Blender)
else
    echo "Could not find a Blender install. Set BLENDER=/path/to/blender and re-run." >&2
    exit 1
fi
echo "==> Using Blender: ${BLENDER_CMD[*]}"

echo "==> Validating extension manifest"
"${BLENDER_CMD[@]}" --command extension validate "$SOURCE_DIR"

echo "==> Building extension package"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
BUILD_ARGS=(--source-dir "$SOURCE_DIR" --output-dir "$BUILD_DIR")
if [ "$SPLIT_PLATFORMS" -eq 1 ]; then
    BUILD_ARGS+=(--split-platforms)
fi
"${BLENDER_CMD[@]}" --command extension build "${BUILD_ARGS[@]}"

ZIP_PATH=""
if [ "$SPLIT_PLATFORMS" -eq 1 ]; then
    ZIP_PATH=$(ls -t "$BUILD_DIR"/*"${PLATFORM//-/_}"*.zip 2>/dev/null | head -n1 || true)
fi
if [ -z "$ZIP_PATH" ]; then
    ZIP_PATH=$(ls -t "$BUILD_DIR"/*.zip | head -n1)
fi
echo "==> Built: $ZIP_PATH"

echo "==> Installing into Blender ($REPO repo)"
"${BLENDER_CMD[@]}" --background --command extension install-file -r "$REPO" -e "$ZIP_PATH"

# Drop any stale bytecode cache left over from the previous install so
# nothing shadows the freshly written .py source.
INSTALLED_DIR=""
for base in \
    "$HOME"/.var/app/org.blender.Blender/config/blender/*/extensions/"$REPO"/"$EXTENSION_ID" \
    "$HOME"/.config/blender/*/extensions/"$REPO"/"$EXTENSION_ID"
do
    if [ -d "$base" ]; then
        INSTALLED_DIR="$base"
        rm -rf "${base:?}/__pycache__"
    fi
done

echo
echo "Done. Installed to: ${INSTALLED_DIR:-<not found -- check the Blender extensions repo path>}"
echo
echo "If Blender is already open, disable and re-enable the JAX-IK add-on"
echo "(Edit > Preferences > Add-ons) to pick up the new code -- reinstalling"
echo "only replaces files on disk, it does not reload an already-running session."

if [ "$RUN_TESTS" -eq 1 ]; then
    echo
    echo "==> Running test suite"
    FAILED=0
    for test_file in "$SCRIPT_DIR"/tests/*.py; do
        name=$(basename "$test_file")
        echo "--- $name ---"
        if ! "${BLENDER_CMD[@]}" --background --factory-startup --python "$test_file"; then
            echo "FAILED: $name"
            FAILED=1
        fi
    done
    echo
    if [ "$FAILED" -eq 1 ]; then
        echo "One or more tests failed." >&2
        exit 1
    fi
    echo "All tests passed."
fi
