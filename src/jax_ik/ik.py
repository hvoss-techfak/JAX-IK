import json
import os
import pathlib
import time
from functools import partial

import configargparse
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from tqdm import tqdm

from jax_ik.helper import (
    compute_sdf,
    deform_mesh,
    load_mesh_data_from_gltf,
    load_mesh_data_from_urdf,
    load_skeleton_from_gltf,
    load_skeleton_from_urdf,
)
from jax_ik.objectives import (
    BoneZeroRotationObj,
    DistanceObjTraj,
    ObjectiveFunction,
    SDFSelfCollisionPenaltyObj,
)

# make cache temp folder
os.makedirs("./jax_cache", exist_ok=True)

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
jax.config.update("jax_platforms", "cpu")


def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample a sequence of frames to a new number of frames using linear interpolation.

    Args:
        data (np.ndarray): The original data array of shape (frames, dim).
        target_frames (int): The desired number of frames after resampling.

    Returns:
        np.ndarray: The resampled data array of shape (target_frames, dim).
    """
    original_frames, dim = data.shape
    if original_frames == target_frames:
        return data.copy()

    original_indices = np.linspace(0.0, 1.0, original_frames)
    target_indices = np.linspace(0.0, 1.0, target_frames)

    resampled_data = np.empty((target_frames, dim), dtype=data.dtype)
    for d in range(dim):
        resampled_data[:, d] = np.interp(target_indices, original_indices, data[:, d])
    return resampled_data


@partial(jax.jit, static_argnums=())
def tf_euler_to_matrix(angles: jnp.ndarray) -> jnp.ndarray:
    """
    Convert XYZ Euler angles (in radians) to a 4x4 homogeneous rotation matrix.

    Args:
        angles (jnp.ndarray): Array of 3 Euler angles [x, y, z] in radians.

    Returns:
        jnp.ndarray: 4x4 rotation matrix.
    """
    cx, cy, cz = jnp.cos(angles)
    sx, sy, sz = jnp.sin(angles)

    # Closed-form for R_z @ R_y @ R_x. Mathematically identical to building
    # the three 4x4 factor matrices (mostly structural zeros/ones) and
    # chaining two 4x4 matmuls, but far cheaper -- and this runs on every
    # forward *and* backward pass of the solver's inner loop.
    return jnp.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz, 0.0],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz, 0.0],
            [-sy, sx * cy, cx * cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )


@partial(jax.jit, static_argnums=())
def tf_matrix_to_euler(R: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a 4x4 rotation matrix to XYZ Euler angles (in radians).

    Args:
        R (jnp.ndarray): 4x4 rotation matrix.

    Returns:
        jnp.ndarray: Array of 3 Euler angles [x, y, z] in radians.
    """
    r31 = R[2, 0]
    angle_y = -jnp.arcsin(jnp.clip(r31, -1.0, 1.0))
    angle_x = jnp.arctan2(R[2, 1], R[2, 2])
    angle_z = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.stack([angle_x, angle_y, angle_z])


@partial(jax.jit, static_argnums=())
def tf_rotation_matrix_from_axis_angle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Create a 4x4 rotation matrix from an axis and angle (right-handed).

    Args:
        axis (jnp.ndarray): 3D axis vector.
        angle (float): Rotation angle in radians.

    Returns:
        jnp.ndarray: 4x4 rotation matrix.
    """
    x, y, z = axis
    c, s, t = jnp.cos(angle), jnp.sin(angle), 1.0 - jnp.cos(angle)

    R3 = jnp.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=jnp.float32,
    )
    R4 = jnp.eye(4, dtype=jnp.float32)
    R4 = R4.at[:3, :3].set(R3)
    return R4


@partial(jax.jit, static_argnums=(3, 5))
def _compute_fk_tf(
    local_array: jnp.ndarray,
    parent_indices: jnp.ndarray,
    default_rotations: jnp.ndarray,
    controlled_indices: tuple,
    angle_vector: jnp.ndarray,
    level_bounds: tuple,
) -> jnp.ndarray:
    """
    Compute forward kinematics for a skeleton given joint angles.

    Bones are stored (by FKSolver._prepare_fk_arrays) ordered by BFS depth
    level, with each level occupying a contiguous slice. Rather than walking
    the hierarchy one bone at a time (N sequential steps), we process one
    depth level at a time as a single batched matmul: every bone in a level
    only depends on bones from strictly shallower levels, which have already
    been resolved. This turns the N-step sequential chain into
    len(level_bounds) steps (the tree depth), which is far shorter for wide,
    shallow skeletons.

    Args:
        local_array (jnp.ndarray): Local bind transforms for each bone (N, 4, 4),
            ordered by depth level.
        parent_indices (jnp.ndarray): Parent indices for each bone (N,), indices
            refer to positions in this same level-sorted ordering.
        default_rotations (jnp.ndarray): Default (identity) rotations for each bone (N, 4, 4).
        controlled_indices (tuple): Indices of controlled bones (level-sorted space).
        angle_vector (jnp.ndarray): Euler angles for controlled bones (K*3,).
        level_bounds (tuple): Static tuple of (start, end) slices, one per depth
            level, into the level-sorted arrays above.

    Returns:
        jnp.ndarray: Global transforms for all bones (N, 4, 4), level-sorted order.
    """
    ctrl_idx_arr = jnp.asarray(controlled_indices, dtype=jnp.int32)
    num_controlled = len(controlled_indices)

    # Compute per-bone rotation matrices from Euler XYZ
    R_updates = jax.vmap(tf_euler_to_matrix)(angle_vector.reshape(num_controlled, 3))

    rotations = default_rotations.at[ctrl_idx_arr].set(R_updates)

    # local_array and rotations don't depend on `carry`, so this combined
    # per-bone transform can be computed once, up front, for every bone in
    # parallel, instead of redoing `local @ rotation` inside each level's
    # step. That leaves only one (data-dependent) matmul per level on the
    # sequential critical path instead of two.
    local_rot = local_array @ rotations

    eye4 = jnp.eye(4, dtype=jnp.float32)
    carry = jnp.zeros_like(local_array)

    # Static Python loop over depth levels (unrolled at trace time): each
    # level is a batched matmul over its bones, with no cross-level data
    # dependency other than the already-resolved `carry`.
    for start, end in level_bounds:
        level_parents = parent_indices[start:end]
        safe_parents = jnp.where(level_parents < 0, 0, level_parents)
        parent_transform = jnp.where(
            (level_parents < 0)[:, None, None],
            eye4[None, :, :],
            carry[safe_parents],
        )
        current = parent_transform @ local_rot[start:end]
        carry = carry.at[start:end].set(current)

    return carry


@register_pytree_node_class
class _ZeroObjective(ObjectiveFunction):
    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    def update_params(self, params_dict):
        pass

    def get_params(self):
        return {}

    def referenced_bones(self):
        return ()

    def __call__(self, X, fk_solver):
        return jnp.float32(0.0)


class _LastFrameFKCache:
    """Wraps an FKSolver so that repeated compute_fk_from_angles(cfg) calls
    with the exact same `cfg` object reuse one cached result, instead of
    recomputing FK (and, critically, redifferentiating through it) once per
    objective that asks for it.

    This matters because JAX's automatic differentiation and XLA's common
    subexpression elimination operate at different stages: if two objectives
    independently call fk_solver.compute_fk_from_angles(x_full[-1]), XLA's
    CSE pass (which runs on the *compiled* graph) can often recognize the
    two forward computations as identical and dedupe them -- but jax.grad
    builds its backward pass *before* that, from the traced program as
    written, so it still produces two separate backward computations
    through FK, one per call site. The only way to also share that backward
    work is to make sure the underlying compute_fk_from_angles call really
    only happens once in the traced program, with its result reused by
    every consumer -- which is what this cache does, for the specific
    (structurally common) case of objectives that only ever need the
    trajectory's last frame; see ObjectiveFunction.LAST_FRAME_ONLY.

    Deliberately created fresh (and thrown away) on every trace of
    compute_objectives below: its cache is scoped to a single forward+
    backward pass, so it can't go stale across retraces or leak between
    them, and it stays fully transparent to any objective that doesn't
    look for it (via __getattr__ passthrough to the wrapped solver).

    Exposes the shared cfg object as `shared_last_frame_cfg` so that an
    objective which sometimes (not unconditionally, so it can't just set
    ObjectiveFunction.LAST_FRAME_ONLY) needs exactly the trajectory's last
    frame can opt into a cache hit too: instead of slicing X itself (which
    would produce a *new* array object -- equal in value, but not the same
    object the cache is keyed on, so it wouldn't hit), it can check for this
    attribute (e.g. via getattr(fk_solver, "shared_last_frame_cfg", None))
    and reuse that exact object. See DistanceObjTraj for the one place this
    is used.
    """

    def __init__(self, solver, x_full):
        self._solver = solver
        self.shared_last_frame_cfg = x_full if x_full.ndim == 1 else x_full[-1]
        self._cached_fk = None

    def compute_fk_from_angles(self, angle_vector):
        if angle_vector is self.shared_last_frame_cfg:
            if self._cached_fk is None:
                self._cached_fk = self._solver.compute_fk_from_angles(angle_vector)
            return self._cached_fk
        return self._solver.compute_fk_from_angles(angle_vector)

    def __getattr__(self, name):
        return getattr(self._solver, name)


_MANDATORY_POOL = []
_OPTIONAL_POOL = []

# Number of initial steps (per solve) where gd_step's cautious mask is
# forced to all-ones instead of the usual (m_hat * grad > 0) gate -- see
# gd_step below. Momentum starts at exactly zero, so the mask's own
# momentum-vs-gradient agreement test is close to arbitrary for the first
# few steps regardless of which direction is actually useful; forcing it
# open lets those early steps make real progress instead of being
# randomly half-vetoed while momentum is still warming up. Swept
# empirically on paper_evaluation/ik_jax_lib_bench_drag.py: 12 gave a
# further ~3% reduction in dist_mean/dist_std over the always-on cautious
# mask, at some extra iterations/wall-clock cost (the mask being open
# longer means more steps before patience-based early stop can trigger).
_MASK_WARMUP = 12


def solve_ik(
    init_rot: np.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
    mandatory_obj_fns: list,
    optional_obj_fns: list,
    fksolver: "FKSolver",
    threshold: float = 0.01,
    num_steps: int = 1000,
    learning_rate: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    patience: int = 200,
    mask: np.ndarray = None,
) -> tuple:
    """
    Solve inverse kinematics using Adam optimizer and a set of objectives.

    Mandatory and optional objectives are both optimized every step (their
    gradients are summed into one combined loss, so optional objectives
    genuinely shape the pose throughout), but only the mandatory objectives'
    summed loss gates convergence and is reported back as the best-loss
    value: once it falls below `threshold`, the solve is considered
    successful and may stop early, regardless of how far optional objectives
    still have to go. If no mandatory objectives are given, this falls back
    to gating on the combined (mandatory + optional) loss instead, matching
    the historical behavior for optional-only solves.

    Args:
        init_rot (np.ndarray): Initial joint angles.
        lower_bounds (jnp.ndarray): Lower joint limits.
        upper_bounds (jnp.ndarray): Upper joint limits.
        mandatory_obj_fns (list): List of mandatory objective functions --
            must converge below `threshold` for the solve to be considered
            successful.
        optional_obj_fns (list): List of optional objective functions --
            still optimized every step, but never required to converge.
        fksolver (FKSolver): Forward kinematics solver.
        threshold (float): Stop once the mandatory objectives' summed loss
            falls below this value (or the combined loss, if there are no
            mandatory objectives).
        num_steps (int): Maximum number of optimization steps.
        learning_rate (float): Adam optimizer learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        epsilon (float): Adam epsilon parameter.
        patience (int): Early stopping patience.
        mask (np.ndarray): Boolean mask for which frames to optimize.

    Returns:
        tuple: (iterations, final_angles, best_loss, status_code) -- best_loss
            is the best mandatory-objective loss seen (or best combined loss,
            if there are no mandatory objectives).
    """
    MAX_MANDATORY = 10
    MAX_OPTIONAL = 10

    global _MANDATORY_POOL, _OPTIONAL_POOL
    if not _MANDATORY_POOL:
        _MANDATORY_POOL = [_ZeroObjective() for _ in range(MAX_MANDATORY)]
    if not _OPTIONAL_POOL:
        _OPTIONAL_POOL = [_ZeroObjective() for _ in range(MAX_OPTIONAL)]

    if len(mandatory_obj_fns) > MAX_MANDATORY:
        raise ValueError(
            f"Maximum {MAX_MANDATORY} mandatory objectives supported, got {len(mandatory_obj_fns)}"
        )
    if len(optional_obj_fns) > MAX_OPTIONAL:
        raise ValueError(
            f"Maximum {MAX_OPTIONAL} optional objectives supported, got {len(optional_obj_fns)}"
        )

    def _populate(pool, caller_fns):
        """
        Update objective functions in the pool. If an objective's type
        changes, it is replaced, triggering a JIT retrace. If only its
        parameters change, it is updated in-place, avoiding a retrace.
        """
        # Update or replace objectives based on the provided list
        for i, new_fn in enumerate(caller_fns):
            if type(pool[i]) is type(new_fn):
                # Same type: update parameters. This modifies the object in the pool.
                # The object's identity remains the same.
                pool[i].update_params(new_fn.get_params())
            else:
                # Different type: replace the object in the pool.
                # This changes the pytree structure, triggering a retrace.
                pool[i] = new_fn

        # Fill the rest of the pool with ZeroObjective
        for i in range(len(caller_fns), len(pool)):
            if not isinstance(pool[i], _ZeroObjective):
                pool[i] = _ZeroObjective()

        return tuple(pool)

    # Whether the caller supplied any *real* mandatory objective, captured
    # before pool-padding (which always yields a fixed 10-slot tuple and
    # would otherwise hide this). Passed through as a static jit argument so
    # _solve_ik_core can gate convergence on mandatory-only loss.
    has_mandatory = len(mandatory_obj_fns) > 0

    static_mandatory = _populate(_MANDATORY_POOL, mandatory_obj_fns)
    static_optional = _populate(_OPTIONAL_POOL, optional_obj_fns)

    if mask is None:
        mask = np.concatenate(
            [np.zeros(init_rot.shape[0] - 1, dtype=bool), np.ones(1, dtype=bool)],
            axis=0,
        )
    else:
        mask = np.asarray(mask, dtype=bool)

    free_indices = np.where(mask)[0].astype(np.int32)

    return _solve_ik_core(
        init_rot,
        lower_bounds,
        upper_bounds,
        static_mandatory,
        static_optional,
        fksolver,
        threshold=threshold,
        num_steps=num_steps,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        patience=patience,
        has_mandatory=has_mandatory,
        free_indices=free_indices,
    )


@partial(
    jax.jit,
    static_argnums=(
        5,  # fksolver
        6,  # threshold
        7,  # num_steps
        8,  # learning_rate
        9,  # beta1
        10,  # beta2
        11,  # epsilon
        12,  # patience
        13,  # has_mandatory
    ),
)
def _solve_ik_core(
    init_rot: jnp.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
    mandatory_obj_fns: tuple,
    optional_obj_fns: tuple,
    fksolver: "FKSolver",
    threshold: float = 0.001,
    num_steps: int = 1000,
    learning_rate: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    patience: int = 200,
    has_mandatory: bool = True,
    free_indices: jnp.ndarray = None,
) -> tuple:
    """
    Core JIT-compiled IK optimization loop using cautious Adam and early stopping.

    Args:
        init_rot (jnp.ndarray): Initial joint angles.
        lower_bounds (jnp.ndarray): Lower joint limits.
        upper_bounds (jnp.ndarray): Upper joint limits.
        mandatory_obj_fns (tuple): Tuple of mandatory objective functions --
            must converge below `threshold` for the solve to be considered
            successful.
        optional_obj_fns (tuple): Tuple of optional objective functions --
            still optimized every step, but never required to converge.
        fksolver (FKSolver): Forward kinematics solver.
        threshold (float): Stop once the mandatory objectives' summed loss
            falls below this value (or the combined loss, if `has_mandatory`
            is False).
        num_steps (int): Maximum number of optimization steps.
        learning_rate (float): Adam optimizer learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        epsilon (float): Adam epsilon parameter.
        patience (int): Early stopping patience.
        has_mandatory (bool): Whether any real (non-padding) mandatory
            objective was supplied. Static: gates whether convergence/best
            selection is judged on mandatory-only loss or combined loss.
        free_indices (jnp.ndarray): Indices of frames to optimize.

    Returns:
        tuple: (iterations, final_angles, best_loss, status_code) -- best_loss
            is the best mandatory-objective loss seen (or best combined loss,
            if `has_mandatory` is False).
    """
    init_rot = jnp.asarray(init_rot, dtype=jnp.float32)
    lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float32)
    upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float32)
    free_indices = jnp.asarray(free_indices, dtype=jnp.int32)  # << NEW

    X_full = init_rot[None, :] if init_rot.ndim == 1 else init_rot

    x0_free = X_full[free_indices]
    free_T = x0_free.shape[0]

    lower_b = jnp.tile(lower_bounds[None, :], (free_T, 1))
    upper_b = jnp.tile(upper_bounds[None, :], (free_T, 1))

    def compute_objectives(x_full):
        # Objectives marked LAST_FRAME_ONLY always reduce X down to
        # `x_full[-1]` internally; routing them through the *same* cfg
        # object plus a shared FK cache lets JAX share the FK
        # forward+backward computation across all of them instead of
        # redoing it once per objective (see _LastFrameFKCache above).
        # Every objective (not just LAST_FRAME_ONLY ones) gets the cache as
        # its fk_solver: it's a transparent passthrough for anything that
        # doesn't ask for shared_last_frame_cfg, but lets an objective whose
        # need for the last frame is data-dependent (e.g. DistanceObjTraj
        # with a single target point) opt into a cache hit too.
        last_frame_fk = _LastFrameFKCache(fksolver, x_full)
        last_cfg = last_frame_fk.shared_last_frame_cfg

        def call(fn):
            if fn.LAST_FRAME_ONLY:
                return fn(last_cfg, last_frame_fk)
            return fn(x_full, last_frame_fk)

        mand = jnp.float32(0.0)
        for fn in mandatory_obj_fns:
            mand = mand + call(fn)
        opt = jnp.float32(0.0)
        for fn in optional_obj_fns:
            opt = opt + call(fn)
        # Stabilize: replace NaN/Inf with large finite sentinel
        mand = jnp.nan_to_num(mand, nan=1e6, posinf=1e6, neginf=1e6)
        opt = jnp.nan_to_num(opt, nan=1e6, posinf=1e6, neginf=1e6)
        total = mand + opt
        total = jnp.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
        return total, mand

    def obj_free(x_free):
        x_full = X_full.at[free_indices].set(x_free)
        return compute_objectives(x_full)

    # has_aux=True: gradient descent still steps on the combined `total`
    # (mand + opt), so optional objectives keep shaping the pose every
    # iteration -- `mand` just rides along from the same forward pass so
    # convergence/best-selection can be judged on it separately, below.
    value_and_grad = jax.value_and_grad(obj_free, has_aux=True)

    # Each iteration needs the objective value *and* gradient both at its
    # starting point (to take the step) and at the resulting point (to
    # decide whether it improved on the best-so-far). Naively that's two
    # forward passes per iteration -- but the "resulting point" of iteration
    # i is the "starting point" of iteration i+1, so its value/gradient only
    # need to be computed once. We thread (total, grad) for the *current* x
    # through the loop state instead of recomputing them at the top of every
    # iteration, cutting one full (FK-heavy) forward pass per step.
    def gd_step(state):
        i, x, m, v, best_x, best_mand, best_total, no_improve, total, mand, grad = state

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * jnp.square(grad)
        # Plain bias-corrected Adam momentum -- not the NAdam-blended
        # numerator (m_hat mixing in a bias-corrected *current* gradient
        # term) this briefly used. That blend directly injects each step's
        # raw, unsmoothed gradient into the step numerator; under multiple
        # *mandatory* objectives that genuinely conflict (e.g. a Reach
        # Target fighting a Zero Rotation and a Prefer Current Pose
        # regularizer, all folded into one combined mandatory loss), the
        # gradient itself oscillates near the resulting trade-off
        # equilibrium, and NAdam's numerator inherited that oscillation
        # directly -- measured as a ~10x larger end-effector position
        # spread under a tiny warm-start perturbation, and visibly jittery
        # solves frame-to-frame in the Blender add-on's Live/Bake usage.
        # Plain momentum smooths that noise out instead of amplifying it.
        # See paper_evaluation/ik_jax_lib_bench_jitter.py.
        m_hat = m / (1.0 - beta1 ** (i + 1))
        v_hat = v / (1.0 - beta2 ** (i + 1))

        # Cautious optimizer modification: mask where momentum and gradient
        # have the same sign -- forced open for the first _MASK_WARMUP
        # steps (see its definition above), since momentum starts at zero
        # and hasn't warmed up enough yet for that agreement test to mean
        # much.
        cautious_mask = (m_hat * grad > 0).astype(grad.dtype)
        mask = jnp.where(i < _MASK_WARMUP, jnp.ones_like(grad), cautious_mask)

        # Apply cautious mask to the normalized gradient
        denom = jnp.sqrt(v_hat) + epsilon
        norm_grad = (m_hat * mask) / denom
        step = learning_rate * norm_grad

        x_new = jnp.clip(x - step, lower_b, upper_b)

        (new_total, new_mand), new_grad = value_and_grad(x_new)

        # "Best" is judged on mandatory loss alone (total as a tie-break)
        # once there's a real mandatory objective, so a step that improves
        # optional at mandatory's expense is never adopted as best; with no
        # mandatory objectives this reduces to the historical total-only
        # comparison.
        if has_mandatory:
            improved = jnp.logical_or(
                new_mand < best_mand,
                jnp.logical_and(new_mand <= best_mand, new_total < best_total),
            )
        else:
            improved = new_total < best_total

        best_x = jax.lax.select(improved, x_new, best_x)
        best_mand = jax.lax.select(improved, new_mand, best_mand)
        best_total = jax.lax.select(improved, new_total, best_total)
        no_improve = jax.lax.select(improved, 0, no_improve + 1)

        return (
            i + 1,
            x_new,
            m,
            v,
            best_x,
            best_mand,
            best_total,
            no_improve,
            new_total,
            new_mand,
            new_grad,
        )

    def gd_cond(state):
        i, x, m, v, best_x, best_mand, best_total, no_improve, total, mand, grad = state
        # Require a minimal number of iterations before allowing threshold-based early stop
        min_thresh_iters = 5
        patience_ret = jnp.logical_and(i < num_steps, no_improve < patience)
        convergence_metric = best_mand if has_mandatory else best_total
        threshold_ret = jnp.logical_or(i < min_thresh_iters, convergence_metric > threshold)
        return jnp.logical_and(patience_ret, threshold_ret)

    (total0, mand0), grad0 = value_and_grad(x0_free)
    init_state = (
        0,
        x0_free,
        jnp.zeros_like(x0_free),
        jnp.zeros_like(x0_free),
        x0_free,
        jnp.inf,
        jnp.inf,
        0,
        total0,
        mand0,
        grad0,
    )
    (
        iterations,
        best_free,
        _,
        _,
        _,
        best_mand,
        best_total,
        _,
        _,
        _,
        _,
    ) = jax.lax.while_loop(gd_cond, gd_step, init_state)

    final_traj = X_full.at[free_indices].set(best_free)
    best_metric = best_mand if has_mandatory else best_total
    return iterations, final_traj, best_metric, jnp.int32(0)


class FKSolver:
    """
    Forward Kinematics solver for a skeleton model.
    Loads skeleton, mesh, and computes SDF if requested.
    """

    def __init__(
        self,
        model_file: str,
        controlled_bones: list = None,
        do_compute_sdf: bool = True,
        bones_of_interest: list = None,
    ):
        """
        Initialize the FKSolver.

        Args:
            model_file (str): Path to the model file (GLB, GLTF, or URDF).
            controlled_bones (list): List of bone names to control.
            compute_sdf (bool): Whether to compute the mesh SDF for collision.
            bones_of_interest (list): Optional. If given, FK is pruned to just
                these bones plus controlled_bones and all of their ancestors
                (every other branch of the skeleton is dropped entirely,
                since nothing on it ever gets queried). Restricting a wide
                skeleton down to the handful of bones on the path an
                objective actually needs can substantially cut FK cost, but
                it means get_bone_head_tail_from_fk (and anything that walks
                fk_solver.bone_names, e.g. self-collision or mesh skinning)
                will only see this pruned bone set -- so leave this None
                (the default: keep the full skeleton) unless you know
                exactly which bones every objective you'll use will query.
        """
        self.model_file = model_file
        self.file_type = pathlib.Path(model_file).suffix.lower()
        self.limits = {}
        self.mesh_data = None
        self.sdf = None

        if self.file_type in [".glb", ".gltf"]:
            self.skeleton = load_skeleton_from_gltf(model_file)
        elif self.file_type == ".urdf":
            self.skeleton, self.limits = load_skeleton_from_urdf(model_file)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        keep_bones = None
        if bones_of_interest is not None:
            keep_bones = self._bone_closure(controlled_bones, bones_of_interest)

        self._finish_init(controlled_bones, keep_bones)

        if do_compute_sdf:
            # Load mesh and compute SDF
            print("Loading mesh for SDF computation...")
            if self.file_type == ".urdf":
                self.mesh_data = load_mesh_data_from_urdf(self.model_file, self)
            else:
                self.mesh_data = load_mesh_data_from_gltf(self.model_file, self)

            if self.mesh_data:
                import trimesh

                print("Computing SDF from mesh...")
                rest_mesh = trimesh.Trimesh(
                    vertices=np.asarray(self.mesh_data["vertices"][:, :3]),
                    faces=np.asarray(self.mesh_data["faces"]),
                )
                self.sdf = compute_sdf(rest_mesh)
                print("SDF computation complete.")
            else:
                print(
                    "Warning: Could not load mesh data. Self-collision will be disabled."
                )

    def _bone_closure(self, controlled_bones: list, extra_bones) -> set:
        """The set of bones to keep when pruning to bones_of_interest:
        controlled_bones + extra_bones, plus every one of their ancestors
        (walking each up to the root). Requires self.skeleton to already be
        loaded.
        """
        wanted = set(controlled_bones or []) | set(extra_bones or [])
        keep_bones = set()
        for name in wanted:
            cur = name
            while cur is not None and cur not in keep_bones:
                keep_bones.add(cur)
                cur = self.skeleton[cur]["parent"]
        return keep_bones

    def _finish_init(self, controlled_bones: list, keep_bones: set) -> None:
        """Shared tail of __init__/_pruned_view: build the FK arrays (with
        optional pruning) and everything derived from them. Assumes
        self.skeleton is already set.
        """
        self._prepare_fk_arrays(keep_bones=keep_bones)
        self.controlled_bones = controlled_bones if controlled_bones is not None else []
        self.controlled_indices = [
            i for i, name in enumerate(self.bone_names) if name in self.controlled_bones
        ]
        self.default_rotations = jnp.stack(
            [jnp.eye(4, dtype=jnp.float32) for _ in self.bone_names], axis=0
        )
        zero_angles = jnp.zeros(len(self.controlled_indices) * 3, dtype=jnp.float32)
        self.bind_fk = self.compute_fk_from_angles(zero_angles)

    @classmethod
    def _pruned_view(
        cls, base: "FKSolver", controlled_bones: list, extra_bones
    ) -> "FKSolver":
        """Build a new FKSolver that reuses `base`'s already-loaded skeleton
        dict (no disk I/O, no mesh/SDF recomputation) but with FK pruned to
        controlled_bones + extra_bones + their ancestors.

        Used by InverseKinematicsSolver to automatically prune FK to just
        the bones a given set of objectives actually reference -- see
        ObjectiveFunction.referenced_bones() and
        InverseKinematicsSolver._fk_solver_for(). The result never has
        mesh_data/sdf (self-collision/mesh objectives declare
        referenced_bones() -> None specifically so they're never routed
        through a pruned view, and instead keep using `base` itself).
        """
        solver = cls.__new__(cls)
        solver.model_file = base.model_file
        solver.file_type = base.file_type
        solver.limits = base.limits
        solver.mesh_data = None
        solver.sdf = None
        solver.skeleton = base.skeleton

        keep_bones = solver._bone_closure(controlled_bones, extra_bones)
        solver._finish_init(controlled_bones, keep_bones)
        return solver

    def _prepare_fk_arrays(self, keep_bones: set = None) -> None:
        """
        Walk the joint hierarchy and create arrays for FK computation.
        Ensures bones are topologically sorted for FK.

        Args:
            keep_bones (set): Optional. If given, only these bones (which
                must already include every one of their ancestors -- see
                the FKSolver constructor, which builds this set) are kept;
                every other branch of the skeleton is skipped entirely.
        """
        self.bone_names = []
        self.local_list = []
        self.parent_list = []

        visited = set()

        def dfs(bone_name, parent_index):
            if bone_name in visited:
                return
            if keep_bones is not None and bone_name not in keep_bones:
                return
            visited.add(bone_name)

            current_idx = len(self.bone_names)
            self.bone_names.append(bone_name)

            bone = self.skeleton[bone_name]
            self.local_list.append(
                jnp.asarray(bone["local_transform"], dtype=jnp.float32)
            )
            self.parent_list.append(parent_index)

            # Process children in consistent order
            for child in sorted(bone["children"]):
                if child in self.skeleton:
                    dfs(child, current_idx)

        # Find root bones (those with no parent)
        roots = [name for name, bone in self.skeleton.items() if bone["parent"] is None]

        # Process roots in consistent order
        for root in sorted(roots):
            dfs(root, -1)

        # Reorder bones by BFS depth level (root=0), stable on the DFS order
        # within a level. Every array/list keyed by bone index (bone_names,
        # local_list, parent_list, ...) is permuted together, so this is
        # transparent to all name-based lookups elsewhere. The payoff is that
        # depth levels become contiguous slices, which lets FK be computed
        # one level at a time (see _compute_fk_tf) instead of one bone at a
        # time.
        n = len(self.bone_names)
        depth = [0] * n
        for i in range(n):
            p = self.parent_list[i]
            depth[i] = 0 if p < 0 else depth[p] + 1

        order = sorted(range(n), key=lambda i: (depth[i], i))
        inv_order = [0] * n
        for new_idx, old_idx in enumerate(order):
            inv_order[old_idx] = new_idx

        self.bone_names = [self.bone_names[i] for i in order]
        self.local_list = [self.local_list[i] for i in order]
        self.parent_list = [
            -1 if self.parent_list[i] < 0 else inv_order[self.parent_list[i]]
            for i in order
        ]

        level_bounds = []
        start = 0
        sorted_depths = [depth[i] for i in order]
        for k in range(1, n + 1):
            if k == n or sorted_depths[k] != sorted_depths[start]:
                level_bounds.append((start, k))
                start = k
        self.level_bounds = tuple(level_bounds)

        # Convert to JAX arrays
        self.local_array = jnp.stack(self.local_list, axis=0)
        self.parent_indices = jnp.asarray(self.parent_list, dtype=jnp.int32)

        print(f"Loaded skeleton with {len(self.bone_names)} bones")
        print(
            f"Root bones: {[name for name, bone in self.skeleton.items() if bone['parent'] is None]}"
        )

        # Debug: Print some bone transforms
        # print("Sample bone local transforms:")
        # for i in range(min(5, len(self.bone_names))):
        #     bone_name = self.bone_names[i]
        #     parent_idx = self.parent_list[i]
        #     parent_name = self.bone_names[parent_idx] if parent_idx >= 0 else "ROOT"
        #     transform = self.local_list[i]
        #     position = transform[:3, 3]
        #     print(f"  {bone_name} (parent: {parent_name}): position = {position}")

    def compute_fk_from_angles(self, angle_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute global bone transforms from provided Euler angles.

        Args:
            angle_vector (jnp.ndarray): Flat array of Euler angles for controlled bones.

        Returns:
            jnp.ndarray: Array of global transforms for all bones.
        """
        angle_vector = jnp.asarray(angle_vector, dtype=jnp.float32)

        result = _compute_fk_tf(
            self.local_array,
            self.parent_indices,
            self.default_rotations,
            tuple(self.controlled_indices),
            angle_vector,
            self.level_bounds,
        )
        return result

    def get_bone_head_tail_from_fk(
        self, fk_transforms: jnp.ndarray, bone_name: str
    ) -> tuple:
        """
        Get the world-space head and tail positions of a bone.

        Args:
            fk_transforms (jnp.ndarray): Array of global transforms for all bones.
            bone_name (str): Name of the bone.

        Returns:
            tuple: (head_position, tail_position) as 1D arrays.
        """
        if bone_name not in self.bone_names:
            print(self.bone_names)
            raise ValueError(f"Bone '{bone_name}' not found in skeleton.")

        idx = self.bone_names.index(bone_name)
        global_transform = fk_transforms[idx]
        head = global_transform[:3, 3]

        bone = self.skeleton[bone_name]
        tail_local = jnp.asarray(
            [0.0, bone["bone_length"], 0.0, 1.0], dtype=jnp.float32
        )
        tail = global_transform @ tail_local
        return head, tail[:3]

    def render(
        self,
        angle_vector: np.ndarray = None,
        target_pos: list = [],
        collider_spheres: list = [],
        mesh_data: dict = None,
        pv_mesh=None,
        interactive: bool = False,
    ) -> None:
        """
        Visualize the skeleton, mesh, and objectives using PyVista.

        Args:
            angle_vector (np.ndarray): Joint angles to render.
            target_pos (list): List of 3D target points to show.
            collider_spheres (list): List of sphere colliders to show.
            mesh_data (dict): Mesh data to use (optional).
            pv_mesh: Existing PyVista mesh object (optional).
            interactive (bool): If True, show interactive window.
        """
        # PyVista (and the VTK it pulls in) is a heavy, rendering-only
        # dependency -- import it lazily so solving IK doesn't pay its
        # import cost/memory unless a caller actually renders.
        import pyvista as pv

        # Prepare angles
        if angle_vector is None:
            angle_vector = jnp.zeros(
                len(self.controlled_indices) * 3, dtype=jnp.float32
            )
        else:
            angle_vector = jnp.asarray(angle_vector, dtype=jnp.float32)

        # FK transforms
        # fk_transforms = self.compute_fk_from_angles(angle_vector)

        # Load mesh data if not provided
        if mesh_data is None:
            if self.file_type == ".urdf":
                mesh_data = load_mesh_data_from_urdf(self.model_file, self)
            else:
                mesh_data = load_mesh_data_from_gltf(self.model_file, self)

        if mesh_data is None:
            print("Cannot render: mesh data is missing.")
            return

        # Deform mesh
        deformed_verts = deform_mesh(angle_vector, self, mesh_data)
        vertices = np.asarray(deformed_verts)
        faces = mesh_data["faces"]
        pv_faces = np.hstack((np.full((faces.shape[0], 1), 3, dtype=int), faces))

        # Create PyVista mesh
        if pv_mesh is None:
            pv_mesh = pv.PolyData(vertices, pv_faces)
        else:
            pv_mesh.points = vertices

        plotter = pv.Plotter()
        plotter.add_mesh(
            pv_mesh, color="lightblue", show_edges=False, smooth_shading=True
        )

        camera_position = [
            0.0,
            0.0,
            3.0,
        ]
        focal_point = [
            0.0,
            0.0,
            0.0,
        ]
        up_vector = [0.0, 1.0, 0.0]  # Y-up orientation

        plotter.camera_position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.up = up_vector

        # Draw target positions
        for pt in target_pos:
            plotter.add_mesh(
                pv.Sphere(radius=0.02, center=np.asarray(pt)), color="green"
            )

        # Draw collider spheres
        for sphere in collider_spheres:
            center = np.asarray(sphere.get("center", [0, 0, 0]))
            radius = float(sphere.get("radius", 0.05))
            plotter.add_mesh(
                pv.Sphere(radius=radius, center=center), color="yellow", opacity=0.5
            )

        plotter.show(title="Skeleton and Deformed Mesh", interactive=interactive)


class InverseKinematicsSolver:
    """
    High-level IK solver that manages FK, bounds, and optimization.
    """

    def __init__(
        self,
        model_file: str,
        controlled_bones: list = None,
        bounds: list = None,
        penalty_weight: float = 0.25,
        threshold: float = 0.01,
        num_steps: int = 1000,
        compute_sdf: bool = True,
    ):
        """
        Initialize the IK solver.

        Args:
            model_file (str): Path to the model file.
            controlled_bones (list): List of bone names to control.
            bounds (list): List of (min, max) tuples for each joint angle (degrees).
            penalty_weight (float): Weight for regularization penalty.
            threshold (float): Stop if loss falls below this value.
            num_steps (int): Maximum number of optimization steps.
            compute_sdf (bool): Whether to compute mesh SDF for collision.
        """
        self.fk_solver = FKSolver(
            model_file=model_file,
            controlled_bones=controlled_bones,
            do_compute_sdf=compute_sdf,
        )
        self.controlled_bones = self.fk_solver.controlled_bones
        # solve()/solve_guess() automatically prune FK down to just the
        # bones the objectives passed to that call actually reference (see
        # _fk_solver_for below) -- cached here so repeated calls with the
        # same set of objective *types* (the common case: only target
        # values change between calls) reuse the same pruned FKSolver
        # instead of rebuilding it, and so JAX only compiles once for it.
        self._pruned_fk_cache = {}

        # Use limits from URDF if available, otherwise use provided bounds
        if self.fk_solver.limits and not bounds:
            print("Using joint limits from URDF file.")
            urdf_bounds = []

            # Load the URDF to get joint information
            if self.fk_solver.file_type == ".urdf":
                import urchin

                robot = urchin.URDF.load(self.fk_solver.model_file, lazy_load_meshes=True)
                joint_info = {}
                for joint in robot.joints:
                    joint_info[joint.child] = {
                        "type": joint.joint_type,
                        "axis": joint.axis
                        if hasattr(joint, "axis") and joint.axis is not None
                        else [0, 0, 1],
                    }

            for bone_name in self.controlled_bones:
                if bone_name in self.fk_solver.limits:
                    lower, upper = self.fk_solver.limits[bone_name]
                    # print(f"Bone '{bone_name}' limits: {lower} to {upper}")

                    # Get joint information
                    if bone_name in joint_info:
                        joint_type = joint_info[bone_name]["type"]
                        joint_axis = joint_info[bone_name]["axis"]

                        if joint_type in ["revolute", "continuous"]:
                            # For revolute joints, apply limits only to the primary rotation axis
                            # Determine which axis has the largest component
                            abs_axis = [abs(x) for x in joint_axis]
                            main_axis = abs_axis.index(max(abs_axis))

                            # Create bounds for X, Y, Z rotations
                            axis_bounds = [
                                (-10, 10),
                                (-10, 10),
                                (-10, 10),
                            ]  # Default small range
                            axis_bounds[main_axis] = (
                                lower,
                                upper,
                            )  # Apply real limits to main axis

                            urdf_bounds.extend(axis_bounds)
                            # print(f"  Applied limits to axis {main_axis}: {axis_bounds}")
                        else:
                            # For other joint types, use conservative limits
                            urdf_bounds.extend([(lower, upper), (-10, 10), (-10, 10)])
                    else:
                        # Default for bones without joint info - conservative limits
                        urdf_bounds.extend([(lower, upper), (-30, 30), (-30, 30)])
                else:
                    # Default for bones without limits
                    urdf_bounds.extend([(-180, 180), (-180, 180), (-180, 180)])
            bounds = urdf_bounds

        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float32)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float32)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps
        self.avg_iter_time = None

    def _fk_solver_for(self, mandatory_objective_functions, optional_objective_functions):
        """Pick the FKSolver to use for a solve() / solve_guess() call:
        self.fk_solver (the full, unpruned skeleton) if any active
        objective needs it -- because it declares
        ObjectiveFunction.referenced_bones() -> None, whether because it
        walks the whole skeleton or depends on fk_solver.sdf/mesh_data/
        bind_fk -- otherwise a cached FKSolver pruned to just
        controlled_bones plus whatever specific bones every objective
        declared it needs.

        This is what makes bones_of_interest-style pruning automatic:
        callers never need to work out or pass which bones are needed
        themselves, and if the set of objectives used from one solve() call
        to the next doesn't change (the common case -- only target values
        differ), the same cached pruned FKSolver is reused, so JAX only
        traces/compiles for it once.
        """
        wanted = set(self.controlled_bones)
        for fn in (*mandatory_objective_functions, *optional_objective_functions):
            bones = fn.referenced_bones()
            if bones is None:
                return self.fk_solver
            wanted.update(bones)

        key = frozenset(wanted)
        fk_solver = self._pruned_fk_cache.get(key)
        if fk_solver is None:
            extra_bones = wanted - set(self.controlled_bones)
            fk_solver = FKSolver._pruned_view(
                self.fk_solver, self.controlled_bones, extra_bones
            )
            self._pruned_fk_cache[key] = fk_solver
        return fk_solver

    def solve_guess(
        self,
        initial_rotations: np.ndarray,
        learning_rate: float = 0.001,
        mandatory_objective_functions: tuple = (),
        optional_objective_functions: tuple = (),
        prefix_len: int = 1,
        patience: int = 200,
    ) -> tuple:
        """
        Solve IK for a trajectory, keeping the first prefix_len frames fixed.

        Args:
            initial_rotations (np.ndarray): Initial joint angle trajectory.
            learning_rate (float): Adam optimizer learning rate.
            mandatory_objective_functions (tuple): Mandatory objectives.
            optional_objective_functions (tuple): Optional objectives.
            prefix_len (int): Number of frames to keep fixed.
            patience (int): Early stopping patience.

        Returns:
            tuple: (final_angles, best_loss, steps)
        """
        X_full = jnp.asarray(initial_rotations, dtype=jnp.float32)
        mask = jnp.concatenate(
            [
                jnp.zeros(prefix_len, dtype=bool),
                jnp.ones(X_full.shape[0] - prefix_len, dtype=bool),
            ]
        )

        steps, best_angles, best_obj, _ = solve_ik(
            init_rot=X_full,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            mandatory_obj_fns=tuple(fn for fn in mandatory_objective_functions),
            optional_obj_fns=tuple(fn for fn in optional_objective_functions),
            fksolver=self._fk_solver_for(
                mandatory_objective_functions, optional_objective_functions
            ),
            threshold=self.threshold,
            num_steps=self.num_steps,
            learning_rate=learning_rate,
            patience=patience,
            mask=mask,
        )
        return np.asarray(best_angles), float(best_obj), int(steps)

    def solve(
        self,
        initial_rotations: np.ndarray = None,
        learning_rate: float = 0.001,
        mandatory_objective_functions: tuple = (),
        optional_objective_functions: tuple = (),
        ik_points: int = 1,
        patience: int = 200,
        verbose: bool = True,
    ) -> tuple:
        """
        Solve IK for a single pose or a short trajectory.

        Args:
            initial_rotations (np.ndarray): Initial joint angles (optional).
            learning_rate (float): Adam optimizer learning rate.
            mandatory_objective_functions (tuple): Mandatory objectives.
            optional_objective_functions (tuple): Optional objectives.
            ik_points (int): Number of frames to optimize after the initial pose.
            patience (int): Early stopping patience.
            verbose (bool): If True, print optimization info.

        Returns:
            tuple: (final_angles, best_loss, steps)
        """
        # Assembled with plain numpy, not jnp, deliberately: each eager jnp
        # call (concatenate/tile/...) is its own dispatch to the XLA
        # runtime, and this array bookkeeping doesn't need a device at all
        # -- solve_ik does a single jnp.asarray on the finished X_full/mask
        # below, so preparing them on the host keeps this call down to one
        # dispatch total instead of several.
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        else:
            initial_rotations = np.asarray(initial_rotations, dtype=np.float32)

        if ik_points < 1:
            ik_points = 1

        if initial_rotations.ndim == 1:
            X_full = np.concatenate(
                [
                    initial_rotations[None, :],
                    np.tile(initial_rotations[None, :], (ik_points, 1)),
                ],
                axis=0,
            )
            mask = np.concatenate(
                [np.array([False]), np.ones(ik_points, dtype=bool)], axis=0
            )
        else:
            T_current = initial_rotations.shape[0]
            extension = np.tile(initial_rotations[-1][None, :], (ik_points, 1))
            X_full = np.concatenate([initial_rotations, extension], axis=0)
            mask = np.concatenate(
                [np.zeros(T_current, dtype=bool), np.ones(ik_points, dtype=bool)],
                axis=0,
            )

        steps, best_angles, best_obj, _ = solve_ik(
            init_rot=X_full,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            mandatory_obj_fns=tuple(fn for fn in mandatory_objective_functions),
            optional_obj_fns=tuple(fn for fn in optional_objective_functions),
            fksolver=self._fk_solver_for(
                mandatory_objective_functions, optional_objective_functions
            ),
            threshold=self.threshold,
            num_steps=self.num_steps,
            learning_rate=learning_rate,
            patience=patience,
            mask=mask,
        )

        if verbose:
            print(f"Optimization took {steps} steps. Best Obj: {best_obj}")
        return np.asarray(best_angles), float(best_obj), int(steps)

    def render(
        self,
        angle_vector: np.ndarray = None,
        target_pos: list = [],
        collider_spheres: list = [],
        mesh_data: dict = None,
        pv_mesh=None,
        interactive: bool = False,
    ) -> None:
        """
        Visualize the current pose and objectives using PyVista.

        Args:
            angle_vector (np.ndarray): Joint angles to render.
            target_pos (list): List of 3D target points to show.
            collider_spheres (list): List of sphere colliders to show.
            mesh_data (dict): Mesh data to use (optional).
            pv_mesh: Existing PyVista mesh object (optional).
            interactive (bool): If True, show interactive window.
        """
        self.fk_solver.render(
            angle_vector=angle_vector,
            target_pos=target_pos,
            collider_spheres=collider_spheres,
            mesh_data=mesh_data,
            pv_mesh=pv_mesh,
            interactive=interactive,
        )


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 or 4x4 rotation matrix to XYZ Euler angles.

    Args:
        R (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Array of 3 Euler angles [x, y, z] in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])


def export_frames(
    initial_rot: np.ndarray,
    solved_angles: np.ndarray,
    controlled_bones: list,
    export_file: str = "ik_frames.json",
) -> None:
    """
    Export a sequence of joint angles to a JSON file.

    Args:
        initial_rot (np.ndarray): Initial joint angles.
        solved_angles (np.ndarray): Final joint angles or trajectory.
        controlled_bones (list): List of bone names.
        export_file (str): Output JSON file path.
    """
    initial_rot = np.asarray(initial_rot)
    num_bones = initial_rot.shape[0] // 3
    if len(controlled_bones) != num_bones:
        raise ValueError("controlled_bones length mismatches initial configuration")

    frames = []
    if solved_angles.ndim == 1:
        frame0 = {
            bone: initial_rot[i * 3 : (i + 1) * 3].tolist()
            for i, bone in enumerate(controlled_bones)
        }
        frame1 = {
            bone: solved_angles[i * 3 : (i + 1) * 3].tolist()
            for i, bone in enumerate(controlled_bones)
        }
        frames.extend([frame0, frame1])
    else:
        for frame in solved_angles:
            frame_dict = {
                bone: frame[i * 3 : (i + 1) * 3].tolist()
                for i, bone in enumerate(controlled_bones)
            }
            frames.append(frame_dict)

    with open(export_file, "w") as f:
        json.dump(frames, f, indent=4)
    print(f"Exported IK frames to {export_file}")


def export_all_frames(
    trajectories: list,
    controlled_bones: list,
    export_file: str = "ik_all_trajectories.json",
) -> None:
    """
    Export multiple trajectories of joint angles to a JSON file.

    Args:
        trajectories (list): List of (initial_rot, solved_angles) tuples.
        controlled_bones (list): List of bone names.
        export_file (str): Output JSON file path.
    """
    all_frames = []
    for init_rot, solved_angles in trajectories:
        init_rot = np.asarray(init_rot)
        num_bones = init_rot.shape[0] // 3
        if len(controlled_bones) != num_bones:
            raise ValueError("controlled_bones length mismatches initial configuration")

        frames = []
        if solved_angles.ndim == 1:
            frame0 = {
                bone: init_rot[i * 3 : (i + 1) * 3].tolist()
                for i, bone in enumerate(controlled_bones)
            }
            frame1 = {
                bone: solved_angles[i * 3 : (i + 1) * 3].tolist()
                for i, bone in enumerate(controlled_bones)
            }
            frames.extend([frame0, frame1])
        else:
            for frame in solved_angles:
                frame_dict = {
                    bone: frame[i * 3 : (i + 1) * 3].tolist()
                    for i, bone in enumerate(controlled_bones)
                }
                frames.append(frame_dict)
        all_frames.extend(frames)

    with open(export_file, "w") as f:
        json.dump(all_frames, f, indent=4)
    print(f"Exported all trajectories to {export_file}")


def compute_objective_breakdown(
    X: np.ndarray, objective_list: list, fk_solver: "FKSolver"
) -> dict:
    """
    Compute the contribution of each objective to the total loss.

    Args:
        X (np.ndarray): Joint angles to evaluate.
        objective_list (list): List of (name, objective_function) tuples.
        fk_solver (FKSolver): Forward kinematics solver.

    Returns:
        dict: Mapping from objective name to loss value.
    """
    X_tensor = jnp.asarray(X, dtype=jnp.float32)
    breakdown = {}
    for name, obj_fn in objective_list:
        contribution = obj_fn(X_tensor, fk_solver)
        numeric = (
            float(contribution)
            if isinstance(contribution, (float, np.number))
            else float(contribution.item())
        )
        breakdown[name] = numeric
    return breakdown


def main() -> None:
    """
    Command-line entry point for running the IK solver.

    Parses arguments, loads the model, sets up objectives, solves IK, and renders results.
    """
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"],
    )
    parser.add(
        "--model_file",
        type=str,
        default="/home/mei/Downloads/robots/pepper_description-master/urdf/pepper.urdf",
        help="Path to the GLB, GLTF, or URDF model file.",
    )
    parser.add(
        "--hand",
        type=str,
        choices=["left", "right"],
        default="left",
        help="For GLTF models, specify hand.",
    )
    parser.add(
        "--bounds",
        type=str,
        default=None,
        help="JSON string for joint bounds, e.g., '[[-10, 10], ...]'",
    )
    parser.add(
        "--controlled_bones",
        type=str,
        default='["LShoulder","LBicep","LForeArm","l_wrist"]',
        help="JSON string of bone names to control.",
    )
    parser.add(
        "--end_effector_bone",
        type=str,
        default="LFinger13_link",
        help="Name of the end-effector bone for the target.",
    )
    parser.add("--threshold", type=float, default=0.005)
    parser.add("--num_steps", type=int, default=10000)
    parser.add(
        "--target_points",
        type=str,
        default=None,
        help="JSON string of target points, e.g., '[[0,0,1], ...]'",
    )
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--additional_objective_weight", type=float, default=0.25)
    parser.add("--subpoints", type=int, default=5)
    parser.add("--render", action="store_true", help="Render the final pose.")
    args = parser.parse_args()

    # Disable GPU for JAX as CPU is a lot faster for this task
    jax.config.update("jax_default_device", "cpu")

    file_type = pathlib.Path(args.model_file).suffix.lower()

    # --- Configuration based on file type ---
    if file_type == ".urdf":
        # For URDF, user must specify controlled bones and end effector
        if not args.controlled_bones or not args.end_effector_bone:
            raise ValueError(
                "For URDF files, --controlled_bones and --end_effector_bone must be provided."
            )
        controlled_bones = json.loads(args.controlled_bones)
        end_effector = args.end_effector_bone
        bounds = json.loads(args.bounds) if args.bounds else None
    else:  # GLTF/GLB
        hand = args.hand
        controlled_bones = [
            f"{hand}_collar",
            f"{hand}_shoulder",
            f"{hand}_elbow",
            f"{hand}_wrist",
        ]
        end_effector = f"{hand}_index3"
        if args.bounds is None:
            bounds = [(-60, 60)] * 3 * len(controlled_bones)  # Default wide bounds
        else:
            bounds = [tuple(b) for b in json.loads(args.bounds)]

    if args.target_points:
        targets = [np.array(p) for p in json.loads(args.target_points)]
    else:
        targets = [np.array([0.3, 0.3, 0.35])]  # Default target

    # --- Initialize Solver ---
    solver = InverseKinematicsSolver(
        args.model_file,
        controlled_bones=controlled_bones,
        bounds=bounds,
        threshold=args.threshold,
        num_steps=args.num_steps,
    )

    print(f"Available bones: {solver.fk_solver.bone_names}")
    print(f"Controlled bones: {solver.controlled_bones}")
    print(f"End effector: {end_effector}")

    # Check if controlled bones and end effector exist
    missing_bones = []
    for bone in controlled_bones:
        if bone not in solver.fk_solver.bone_names:
            missing_bones.append(bone)
    if end_effector not in solver.fk_solver.bone_names:
        missing_bones.append(end_effector)

    if missing_bones:
        print(
            f"Error: The following bones were not found in the skeleton: {missing_bones}"
        )
        print("Please check the bone names and update the configuration.")
        return

    initial_rotations = np.zeros(len(solver.controlled_bones) * 3, dtype=np.float32)
    final_angles = initial_rotations.copy()

    # Show initial pose
    print("Rendering initial pose...")
    solver.render(angle_vector=final_angles, target_pos=targets, interactive=True)

    # --- Solve for Targets ---
    start_time = time.time()
    for i, target in enumerate(tqdm(targets, desc="Solving IK")):
        mandatory_obj_fns = [
            DistanceObjTraj(
                target_points=[target],
                bone_name=end_effector,
                use_head=True,
                weight=1.0,
            )
        ]
        optional_obj_fns = [
            BoneZeroRotationObj(weight=args.additional_objective_weight),
            SDFSelfCollisionPenaltyObj(
                bone_names=controlled_bones,
                num_samples_per_bone=5,
                min_dist=0.02,
                weight=1.0,
            ),
        ]

        best_angles, obj, steps = solver.solve(
            initial_rotations=final_angles,
            learning_rate=args.learning_rate,
            mandatory_objective_functions=mandatory_obj_fns,
            optional_objective_functions=optional_obj_fns,
            ik_points=args.subpoints,
            verbose=False,
        )
        final_angles = best_angles[-1]
        print(f"Target {i} solved in {steps} steps. Objective: {obj:.4f}")

    total_time = time.time() - start_time
    print(f"\nSolved for {len(targets)} targets in {total_time:.2f} seconds.")

    # --- Render Final Pose ---
    print("Rendering final pose...")
    solver.render(angle_vector=final_angles, target_pos=targets, interactive=True)


if __name__ == "__main__":
    main()
