import json
import os
import time
from functools import partial

import configargparse
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
import pyvista as pv
from tqdm import tqdm

from IK_Helper import deform_mesh as deform_mesh_jax, load_mesh_data_from_gltf


from IK_Helper import load_skeleton_from_gltf

from IK_objectives_jax import (
    DistanceObjTraj,
    ObjectiveFunction, BoneZeroRotationObj,
)

#make cache temp folder
os.makedirs("./jax_cache", exist_ok=True)

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
jax.config.update("jax_platforms", "cpu")


def resample_frames(data, target_frames):
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
def tf_euler_to_matrix(angles):
    """Convert XYZ Euler angles (radians) to a 4×4 homogeneous rotation matrix."""
    cx, cy, cz = jnp.cos(angles)
    sx, sy, sz = jnp.sin(angles)

    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cx, -sx, 0.0],
            [0.0, sx, cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    R_y = jnp.array(
        [
            [cy, 0.0, sy, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sy, 0.0, cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    R_z = jnp.array(
        [
            [cz, -sz, 0.0, 0.0],
            [sz, cz, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    return R_z @ R_y @ R_x


@partial(jax.jit, static_argnums=())
def tf_matrix_to_euler(R):
    """Inverse of tf_euler_to_matrix – returns XYZ Euler angles (radians)."""
    r31 = R[2, 0]
    angle_y = -jnp.arcsin(jnp.clip(r31, -1.0, 1.0))
    angle_x = jnp.arctan2(R[2, 1], R[2, 2])
    angle_z = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.stack([angle_x, angle_y, angle_z])


@partial(jax.jit, static_argnums=())
def tf_rotation_matrix_from_axis_angle(axis, angle):
    """Axis-angle (right-handed) to homogeneous 4×4 matrix."""
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


@partial(jax.jit, static_argnums=(3,))
def _compute_fk_tf(
    local_array: jnp.ndarray,  # (N,4,4)  – bind transforms
    parent_indices: jnp.ndarray,  # (N,)     – parents (-1 = root)
    default_rotations: jnp.ndarray,  # (N,4,4)  – identity for most bones
    controlled_indices: tuple,  # *** tuple of ints – hashable! ***
    angle_vector: jnp.ndarray,  # (K*3,)   – Euler angles for the K controlled bones
) -> jnp.ndarray:
    """
    Vectorised forward-kinematics with a hashable static skeleton layout.
    """
    ctrl_idx_arr = jnp.asarray(controlled_indices, dtype=jnp.int32)
    num_controlled = len(controlled_indices)

    # Compute per-bone rotation matrices from Euler XYZ
    R_updates = jax.vmap(tf_euler_to_matrix)(angle_vector.reshape(num_controlled, 3))

    rotations = default_rotations.at[ctrl_idx_arr].set(R_updates)

    n_bones = local_array.shape[0]
    eye4 = jnp.eye(4, dtype=jnp.float32)

    # Forward pass through the hierarchy
    def fk_body(carry, idx):
        parent_transform = jax.lax.cond(
            parent_indices[idx] < 0,
            lambda _: eye4,
            lambda p: carry[p],
            operand=parent_indices[idx],
        )
        current = parent_transform @ local_array[idx] @ rotations[idx]
        carry = carry.at[idx].set(current)
        return carry, None

    init = jnp.zeros_like(local_array)
    out, _ = jax.lax.scan(fk_body, init, jnp.arange(n_bones))
    return out


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

    def __call__(self, X, fk_solver):
        return jnp.float32(0.0)


_MANDATORY_POOL = []
_OPTIONAL_POOL = []


def solve_ik(
    init_rot,
    lower_bounds,
    upper_bounds,
    mandatory_obj_fns,
    optional_obj_fns,
    fksolver,
    threshold=0.01,
    num_steps=1000,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    patience=200,
    mask=None,
):
    """
    Wraps _solve_ik_core with static objective pools so JIT recompilation
    never occurs between calls (mirrors the TensorFlow trick).
    """
    MAX_MANDATORY = 10
    MAX_OPTIONAL = 10

    global _MANDATORY_POOL, _OPTIONAL_POOL
    if not _MANDATORY_POOL:
        _MANDATORY_POOL = [_ZeroObjective() for _ in range(MAX_MANDATORY)]
    if not _OPTIONAL_POOL:
        _OPTIONAL_POOL = [_ZeroObjective() for _ in range(MAX_OPTIONAL)]

    if len(mandatory_obj_fns) > MAX_MANDATORY:
        raise ValueError(f"Maximum {MAX_MANDATORY} mandatory objectives supported, got {len(mandatory_obj_fns)}")
    if len(optional_obj_fns) > MAX_OPTIONAL:
        raise ValueError(f"Maximum {MAX_OPTIONAL} optional objectives supported, got {len(optional_obj_fns)}")

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
        free_indices=free_indices,  # << NEW
        mask=mask,  # keep for shape checks
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
    ),
)
def _solve_ik_core(
    init_rot,
    lower_bounds,
    upper_bounds,
    mandatory_obj_fns,
    optional_obj_fns,
    fksolver,
    threshold=0.01,
    num_steps=1000,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    patience=200,
    free_indices=None,
    mask=None,
):

    init_rot = jnp.asarray(init_rot, dtype=jnp.float32)
    lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float32)
    upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float32)
    free_indices = jnp.asarray(free_indices, dtype=jnp.int32)  # << NEW

    X_full = init_rot[None, :] if init_rot.ndim == 1 else init_rot
    T_total = X_full.shape[0]

    if mask is None:
        mask = jnp.concatenate([jnp.zeros(T_total - 1, dtype=bool), jnp.ones(1, dtype=bool)], axis=0)
    mask = jnp.asarray(mask, dtype=bool)

    x0_free = X_full[free_indices]
    free_T = x0_free.shape[0]

    lower_b = jnp.tile(lower_bounds[None, :], (free_T, 1))
    upper_b = jnp.tile(upper_bounds[None, :], (free_T, 1))


    def compute_objectives(x_full):
        mand = jnp.float32(0.0)
        for fn in mandatory_obj_fns:
            mand += fn(x_full, fksolver)
        opt = jnp.float32(0.0)
        for fn in optional_obj_fns:
            opt += fn(x_full, fksolver)
        return mand + opt, mand, opt

    def obj_free(x_free):
        x_full = X_full.at[free_indices].set(x_free)
        return compute_objectives(x_full)

    value_and_grad = jax.value_and_grad(lambda x: obj_free(x)[0])


    def gd_step(state):
        i, x, m, v, best_x, best_total, best_mand, best_opt, no_improve = state

        total, grad = value_and_grad(x)
        mand, opt = obj_free(x)[1:]

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * jnp.square(grad)
        m_hat = m / (1.0 - beta1 ** (i + 1))
        v_hat = v / (1.0 - beta2 ** (i + 1))

        # Cautious optimizer modification
        # Create mask where exponential moving average and gradient have same sign
        mask = (m * grad > 0).astype(grad.dtype)
        # Normalize mask by its mean, clamped to avoid division by very small numbers
        mask = mask / jnp.maximum(mask.mean(), 1e-3)
        
        # Apply cautious mask to the normalized gradient
        denom = jnp.sqrt(v_hat) + epsilon
        norm_grad = (m_hat * mask) / denom
        step = learning_rate * norm_grad
        
        x_new = jnp.clip(x - step, lower_b, upper_b)

        new_total, new_mand, new_opt = obj_free(x_new)
        improved = new_total < best_total

        best_x = jax.lax.select(improved, x_new, best_x)
        best_total = jnp.minimum(new_total, best_total)
        best_mand = jnp.minimum(new_mand, best_mand)
        best_opt = jnp.minimum(new_opt, best_opt)
        no_improve = jax.lax.select(improved, 0, no_improve + 1)

        return (
            i + 1,
            x_new,
            m,
            v,
            best_x,
            best_total,
            best_mand,
            best_opt,
            no_improve,
        )

    def gd_cond(state):
        i, x, m, v, best_x, best_total, best_mand, best_opt, no_improve = state
        # ← exactly the same stop-condition TensorFlow used
        patience_ret = jnp.logical_and(i < num_steps, no_improve < patience)
        threshold_ret = best_total > threshold
        return jnp.logical_and(patience_ret, threshold_ret)

    init_state = (
        0,
        x0_free,
        jnp.zeros_like(x0_free),
        jnp.zeros_like(x0_free),
        x0_free,
        jnp.inf,
        jnp.inf,
        jnp.inf,
        0,
    )
    (
        iterations,
        best_free,
        _,
        _,
        _,
        best_total,
        _,
        _,
        _,
    ) = jax.lax.while_loop(gd_cond, gd_step, init_state)


    final_traj = X_full.at[free_indices].set(best_free)
    return iterations, final_traj, best_total, jnp.int32(0)


class FKSolver:
    def __init__(self, gltf_file, controlled_bones=None):
        self.skeleton = load_skeleton_from_gltf(gltf_file)
        self._prepare_fk_arrays()
        self.controlled_bones = controlled_bones if controlled_bones is not None else []
        self.controlled_indices = [i for i, name in enumerate(self.bone_names) if name in self.controlled_bones]
        self.default_rotations = jnp.stack([jnp.eye(4, dtype=jnp.float32) for _ in self.bone_names], axis=0)
        controlled_map = -np.ones(len(self.bone_names), dtype=np.int32)
        for j, bone_idx in enumerate(self.controlled_indices):
            controlled_map[bone_idx] = 3 * j
        self.controlled_map_array = jnp.asarray(controlled_map, dtype=jnp.int32)
        zero_angles = jnp.zeros(len(self.controlled_indices) * 3, dtype=jnp.float32)
        self.bind_fk = self.compute_fk_from_angles(zero_angles)

    def _prepare_fk_arrays(self):
        """
        Walk the joint hierarchy once and create:

          • self.bone_names         list(str)        names in topological order
          • self.local_list         list[(4,4)]      local bind transforms
          • self.parent_list        list[int]        Python list of parents
          • self.local_array        jnp.ndarray(N,4,4)  stacked local matrices
          • self.parent_indices     jnp.ndarray(N,)     same as parent_list but int32
        """
        self.bone_names = []
        self.local_list = []  # keep for objectives that expect the Python list
        self.parent_list = []  # keep for objectives that iterate over parents

        def dfs(bone_name, parent_index):
            current_idx = len(self.bone_names)
            self.bone_names.append(bone_name)

            bone = self.skeleton[bone_name]
            # homogeneous 4×4 bind transform
            self.local_list.append(jnp.asarray(bone["local_transform"], dtype=jnp.float32))
            self.parent_list.append(parent_index)

            for child in bone["children"]:
                dfs(child, current_idx)

        # One DFS per root so multiple‐root skeletons are supported
        roots = [b["name"] for b in self.skeleton.values() if b["parent"] is None]
        for root in roots:
            dfs(root, -1)

        # NumPy-style arrays for fast JAX use
        self.local_array = jnp.stack(self.local_list, axis=0)  # (N,4,4)
        self.parent_indices = jnp.asarray(self.parent_list, dtype=jnp.int32)  # (N,)

    def compute_fk_from_angles(self, angle_vector):
        """
        Compute global bone transforms from the provided Euler angles.
        """
        angle_vector = jnp.asarray(angle_vector, dtype=jnp.float32)

        return _compute_fk_tf(
            self.local_array,
            self.parent_indices,
            self.default_rotations,
            tuple(self.controlled_indices),  # *** tuple – hashable static arg ***
            angle_vector,
        )

    def get_bone_head_tail_from_fk(self, fk_transforms, bone_name):
        if bone_name not in self.bone_names:
            print(self.bone_names)
            raise ValueError(f"Bone '{bone_name}' not found in skeleton.")

        idx = self.bone_names.index(bone_name)
        global_transform = fk_transforms[idx]
        head = global_transform[:3, 3]

        bone = self.skeleton[bone_name]
        tail_local = jnp.asarray([0.0, bone["bone_length"], 0.0, 1.0], dtype=jnp.float32)
        tail = global_transform @ tail_local
        return head, tail[:3]

    def render(
        self,
        angle_vector=None,
        target_pos=[],
        collider_spheres=[],
        mesh_data=None,
        pv_mesh=None,
        interactive=False,
    ):
        """
        Visualize the skeleton, mesh, and objectives using PyVista.
        """
        # Prepare angles
        if angle_vector is None:
            angle_vector = jnp.zeros(len(self.controlled_indices) * 3, dtype=jnp.float32)
        else:
            angle_vector = jnp.asarray(angle_vector, dtype=jnp.float32)

        # FK transforms
        fk_transforms = self.compute_fk_from_angles(angle_vector)

        # Load mesh data if not provided
        if mesh_data is None:
            mesh_data = load_mesh_data_from_gltf(self.skeleton, self)
        # Deform mesh
        deformed_verts = deform_mesh_jax(angle_vector, self, mesh_data)
        vertices = np.asarray(deformed_verts)
        faces = mesh_data["faces"]
        pv_faces = np.hstack((np.full((faces.shape[0], 1), 3, dtype=int), faces))

        # Create PyVista mesh
        if pv_mesh is None:
            pv_mesh = pv.PolyData(vertices, pv_faces)
        else:
            pv_mesh.points = vertices

        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="lightblue", show_edges=False, smooth_shading=True)

        # Draw target positions
        for pt in target_pos:
            plotter.add_mesh(pv.Sphere(radius=0.02, center=np.asarray(pt)), color="green")

        # Draw collider spheres
        for sphere in collider_spheres:
            center = np.asarray(sphere.get("center", [0, 0, 0]))
            radius = float(sphere.get("radius", 0.05))
            plotter.add_mesh(pv.Sphere(radius=radius, center=center), color="yellow", opacity=0.5)

        plotter.show(title="Skeleton and Deformed Mesh", interactive=interactive)



class InverseKinematicsSolver:
    def __init__(
        self,
        gltf_file,
        controlled_bones=None,
        bounds=None,
        penalty_weight=0.25,
        threshold=0.01,
        num_steps=1000,
    ):
        self.fk_solver = FKSolver(gltf_file=gltf_file, controlled_bones=controlled_bones)
        self.controlled_bones = self.fk_solver.controlled_bones

        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float32)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float32)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps
        self.avg_iter_time = None

    def solve_guess(
        self,
        initial_rotations,
        learning_rate=0.2,
        mandatory_objective_functions=(),
        optional_objective_functions=(),
        prefix_len=1,
        patience=200,
    ):
        X_full = jnp.asarray(initial_rotations, dtype=jnp.float32)
        mask = jnp.concatenate([jnp.zeros(prefix_len, dtype=bool), jnp.ones(X_full.shape[0] - prefix_len, dtype=bool)])

        steps, best_angles, best_obj, _ = solve_ik(
            init_rot=X_full,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            mandatory_obj_fns=tuple(fn for fn in mandatory_objective_functions),
            optional_obj_fns=tuple(fn for fn in optional_objective_functions),
            fksolver=self.fk_solver,
            threshold=self.threshold,
            num_steps=self.num_steps,
            learning_rate=learning_rate,
            patience=patience,
            mask=mask,
        )
        return np.asarray(best_angles), float(best_obj), int(steps)

    def solve(
        self,
        initial_rotations=None,
        learning_rate=0.2,
        mandatory_objective_functions=(),
        optional_objective_functions=(),
        ik_points=1,
        patience=200,
        verbose=True,
    ):
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        initial_rotations = jnp.asarray(initial_rotations, dtype=jnp.float32)

        if ik_points < 1:
            ik_points = 1

        if initial_rotations.ndim == 1:
            X_full = jnp.concatenate(
                [
                    initial_rotations[None, :],
                    jnp.tile(initial_rotations[None, :], (ik_points, 1)),
                ],
                axis=0,
            )
            mask = jnp.concatenate([jnp.array([False]), jnp.ones(ik_points, dtype=bool)], axis=0)
        else:
            T_current = initial_rotations.shape[0]
            extension = jnp.tile(initial_rotations[-1][None, :], (ik_points, 1))
            X_full = jnp.concatenate([initial_rotations, extension], axis=0)
            mask = jnp.concatenate([jnp.zeros(T_current, dtype=bool), jnp.ones(ik_points, dtype=bool)], axis=0)

        steps, best_angles, best_obj, _ = solve_ik(
            init_rot=X_full,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            mandatory_obj_fns=tuple(fn for fn in mandatory_objective_functions),
            optional_obj_fns=tuple(fn for fn in optional_objective_functions),
            fksolver=self.fk_solver,
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
        angle_vector=None,
        target_pos=[],
        collider_spheres=[],
        mesh_data=None,
        pv_mesh=None,
        interactive=False,
    ):
        self.fk_solver.render(
            angle_vector=angle_vector,
            target_pos=target_pos,
            collider_spheres=collider_spheres,
            mesh_data=mesh_data,
            pv_mesh=pv_mesh,
            interactive=interactive,
        )

def matrix_to_euler_xyz(R):
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


def export_frames(initial_rot, solved_angles, controlled_bones, export_file="ik_frames.json"):
    initial_rot = np.asarray(initial_rot)
    num_bones = initial_rot.shape[0] // 3
    if len(controlled_bones) != num_bones:
        raise ValueError("controlled_bones length mismatches initial configuration")

    frames = []
    if solved_angles.ndim == 1:
        frame0 = {bone: initial_rot[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
        frame1 = {bone: solved_angles[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
        frames.extend([frame0, frame1])
    else:
        for frame in solved_angles:
            frame_dict = {bone: frame[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
            frames.append(frame_dict)

    with open(export_file, "w") as f:
        json.dump(frames, f, indent=4)
    print(f"Exported IK frames to {export_file}")


def export_all_frames(trajectories, controlled_bones, export_file="ik_all_trajectories.json"):
    all_frames = []
    for init_rot, solved_angles in trajectories:
        init_rot = np.asarray(init_rot)
        num_bones = init_rot.shape[0] // 3
        if len(controlled_bones) != num_bones:
            raise ValueError("controlled_bones length mismatches initial configuration")

        frames = []
        if solved_angles.ndim == 1:
            frame0 = {bone: init_rot[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
            frame1 = {bone: solved_angles[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
            frames.extend([frame0, frame1])
        else:
            for frame in solved_angles:
                frame_dict = {bone: frame[i * 3 : (i + 1) * 3].tolist() for i, bone in enumerate(controlled_bones)}
                frames.append(frame_dict)
        all_frames.extend(frames)

    with open(export_file, "w") as f:
        json.dump(all_frames, f, indent=4)
    print(f"Exported all trajectories to {export_file}")


def compute_objective_breakdown(X, objective_list, fk_solver):
    X_tensor = jnp.asarray(X, dtype=jnp.float32)
    breakdown = {}
    for name, obj_fn in objective_list:
        contribution = obj_fn(X_tensor, fk_solver)
        numeric = float(contribution) if isinstance(contribution, (float, np.number)) else float(contribution.item())
        breakdown[name] = numeric
    return breakdown

def main():
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"],
    )
    parser.add("--gltf_file", type=str, default="smplx.glb")
    parser.add("--hand", type=str, choices=["left", "right"], default="left")
    parser.add("--bounds", type=str, default=None)
    parser.add("--controlled_bones", type=str, default=None)
    parser.add("--threshold", type=float, default=0.005)
    parser.add("--num_steps", type=int, default=10000)
    parser.add("--target_points", type=str, default=None)
    parser.add("--learning_rate", type=float, default=0.1)
    parser.add("--additional_objective_weight", type=float, default=0.25)
    parser.add("--subpoints", type=int, default=5)
    args = parser.parse_args()

    # Disable GPU for JAX as CPU is a lot faster for this task
    jax.config.update("jax_default_device", "cpu")

    hand = args.hand

    if args.bounds is None:
        if hand == "left":
            bounds = [
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (-60, 25),
                (-140, 50),
                (-70, 25),
                (-90, 45),
                (-180, 5),
                (-10, 10),
                (-90, 90),
                (-70, 70),
                (-80, 80),
            ]
        else:
            bounds = [
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (-60, 25),
                (-50, 140),
                (-25, 70),
                (-90, 45),
                (-5, 180),
                (-10, 10),
                (-90, 90),
                (-70, 70),
                (-80, 80),
            ]
    else:
        bounds = [tuple(b) for b in json.loads(args.bounds)]
    # fmt: off

    controlled_bones = [f"{hand}_collar",f"{hand}_shoulder",f"{hand}_elbow",f"{hand}_wrist",]
    targets = [np.array([0.0, 0.2, 0.35]),np.array([0.3, 0.3, 0.35]),np.array([0.3, -0.3, 0.5])]

    solver = InverseKinematicsSolver(
        args.gltf_file,
        controlled_bones=controlled_bones,
        bounds=bounds,
        threshold=args.threshold,
        num_steps=args.num_steps,
    )
    initial_rotations = np.zeros(len(solver.controlled_bones) * 3, dtype=np.float32)
    step_list = []
    time_list = []
    for k in tqdm(range(110)):
        for i, target in enumerate(targets):
        
            mandatory_obj_fns = [DistanceObjTraj(target_points=[target,], bone_name=f"{hand}_index3", use_head=True, weight=1.0)]
            optional_obj_fns = [BoneZeroRotationObj(weight=0.1)]
            if k > 10:
                time1 = time.time()
        
            best_angles, obj, steps = solver.solve(
                initial_rotations=initial_rotations,
                learning_rate=args.learning_rate,
                mandatory_objective_functions=mandatory_obj_fns,
                optional_objective_functions=optional_obj_fns,
                ik_points=args.subpoints,
            )
            if k > 10:
                time_list.append(time.time() - time1)
                step_list.append(steps)
        
            # #print(f"Solving for target {i}: {target}")
            # 
            # if best_angles.ndim == 1:
            #     traj = np.stack([initial_rotations, best_angles], axis=0)
            # else:
            #     traj = best_angles
            # export_data.append((initial_rotations.copy(), traj))
            # initial_rotations = traj[-1]
            #print(f"Target {i} solved – objective {obj:.6f}, steps {steps}")
            
    print(f"Average steps per target: {np.mean(step_list):.2f} ± {np.std(step_list):.2f}")
    print(f"Total time for {len(targets)} targets: {np.sum(time_list):.2f} seconds")
    print(f"Median time per target: {np.median(time_list):.2f} ± {np.std(time_list):.2f} seconds")
        
if __name__ == "__main__":
    main()
