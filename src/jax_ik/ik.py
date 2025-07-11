import json
import os
import time
from functools import partial
import pathlib

import configargparse
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
import pyvista as pv
from tqdm import tqdm

from jax_ik.helper import deform_mesh, load_mesh_data_from_gltf, load_mesh_data_from_urdf, compute_sdf


from jax_ik.helper import load_skeleton_from_gltf, load_skeleton_from_urdf

from jax_ik.objectives import (
    DistanceObjTraj,
    ObjectiveFunction, BoneZeroRotationObj, SDFSelfCollisionPenaltyObj,
)

#make cache temp folder
os.makedirs("./jax_cache", exist_ok=True)

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
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


@partial(jax.jit, static_argnums=(3,))
def _compute_fk_tf(
    local_array: jnp.ndarray,
    parent_indices: jnp.ndarray,
    default_rotations: jnp.ndarray,
    controlled_indices: tuple,
    angle_vector: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute forward kinematics for a skeleton given joint angles.

    Args:
        local_array (jnp.ndarray): Local bind transforms for each bone (N, 4, 4).
        parent_indices (jnp.ndarray): Parent indices for each bone (N,).
        default_rotations (jnp.ndarray): Default (identity) rotations for each bone (N, 4, 4).
        controlled_indices (tuple): Indices of controlled bones.
        angle_vector (jnp.ndarray): Euler angles for controlled bones (K*3,).

    Returns:
        jnp.ndarray: Global transforms for all bones (N, 4, 4).
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

    Args:
        init_rot (np.ndarray): Initial joint angles.
        lower_bounds (jnp.ndarray): Lower joint limits.
        upper_bounds (jnp.ndarray): Upper joint limits.
        mandatory_obj_fns (list): List of mandatory objective functions.
        optional_obj_fns (list): List of optional objective functions.
        fksolver (FKSolver): Forward kinematics solver.
        threshold (float): Stop if loss falls below this value.
        num_steps (int): Maximum number of optimization steps.
        learning_rate (float): Adam optimizer learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        epsilon (float): Adam epsilon parameter.
        patience (int): Early stopping patience.
        mask (np.ndarray): Boolean mask for which frames to optimize.

    Returns:
        tuple: (iterations, final_angles, best_loss, status_code)
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
    ),
)
def _solve_ik_core(
    init_rot: jnp.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
    mandatory_obj_fns: tuple,
    optional_obj_fns: tuple,
    fksolver: "FKSolver",
    threshold: float = 0.01,
    num_steps: int = 1000,
    learning_rate: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    patience: int = 200,
    free_indices: jnp.ndarray = None,
) -> tuple:
    """
    Core JIT-compiled IK optimization loop using cautious Adam and early stopping.

    Args:
        init_rot (jnp.ndarray): Initial joint angles.
        lower_bounds (jnp.ndarray): Lower joint limits.
        upper_bounds (jnp.ndarray): Upper joint limits.
        mandatory_obj_fns (tuple): Tuple of mandatory objective functions.
        optional_obj_fns (tuple): Tuple of optional objective functions.
        fksolver (FKSolver): Forward kinematics solver.
        threshold (float): Stop if loss falls below this value.
        num_steps (int): Maximum number of optimization steps.
        learning_rate (float): Adam optimizer learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        epsilon (float): Adam epsilon parameter.
        patience (int): Early stopping patience.
        free_indices (jnp.ndarray): Indices of frames to optimize.

    Returns:
        tuple: (iterations, final_angles, best_loss, status_code)
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
    """
    Forward Kinematics solver for a skeleton model.
    Loads skeleton, mesh, and computes SDF if requested.
    """

    def __init__(self, model_file: str, controlled_bones: list = None, do_compute_sdf: bool = True):
        """
        Initialize the FKSolver.

        Args:
            model_file (str): Path to the model file (GLB, GLTF, or URDF).
            controlled_bones (list): List of bone names to control.
            compute_sdf (bool): Whether to compute the mesh SDF for collision.
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
                    faces=np.asarray(self.mesh_data["faces"])
                )
                self.sdf = compute_sdf(rest_mesh)
                print("SDF computation complete.")
            else:
                print("Warning: Could not load mesh data. Self-collision will be disabled.")

    def _prepare_fk_arrays(self) -> None:
        """
        Walk the joint hierarchy and create arrays for FK computation.
        Ensures bones are topologically sorted for FK.
        """
        self.bone_names = []
        self.local_list = []
        self.parent_list = []

        visited = set()

        def dfs(bone_name, parent_index):
            if bone_name in visited:
                return
            visited.add(bone_name)

            current_idx = len(self.bone_names)
            self.bone_names.append(bone_name)

            bone = self.skeleton[bone_name]
            self.local_list.append(jnp.asarray(bone["local_transform"], dtype=jnp.float32))
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

        # Convert to JAX arrays
        self.local_array = jnp.stack(self.local_list, axis=0)
        self.parent_indices = jnp.asarray(self.parent_list, dtype=jnp.int32)

        print(f"Loaded skeleton with {len(self.bone_names)} bones")
        print(f"Root bones: {[name for name, bone in self.skeleton.items() if bone['parent'] is None]}")

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
        )
        return result

    def get_bone_head_tail_from_fk(self, fk_transforms: jnp.ndarray, bone_name: str) -> tuple:
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
        tail_local = jnp.asarray([0.0, bone["bone_length"], 0.0, 1.0], dtype=jnp.float32)
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
        # Prepare angles
        if angle_vector is None:
            angle_vector = jnp.zeros(len(self.controlled_indices) * 3, dtype=jnp.float32)
        else:
            angle_vector = jnp.asarray(angle_vector, dtype=jnp.float32)

        # FK transforms
        #fk_transforms = self.compute_fk_from_angles(angle_vector)

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
        plotter.add_mesh(pv_mesh, color="lightblue", show_edges=False, smooth_shading=True)

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
            plotter.add_mesh(pv.Sphere(radius=0.02, center=np.asarray(pt)), color="green")

        # Draw collider spheres
        for sphere in collider_spheres:
            center = np.asarray(sphere.get("center", [0, 0, 0]))
            radius = float(sphere.get("radius", 0.05))
            plotter.add_mesh(pv.Sphere(radius=radius, center=center), color="yellow", opacity=0.5)

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
        self.fk_solver = FKSolver(model_file=model_file, controlled_bones=controlled_bones,do_compute_sdf=compute_sdf)
        self.controlled_bones = self.fk_solver.controlled_bones

        # Use limits from URDF if available, otherwise use provided bounds
        if self.fk_solver.limits and not bounds:
            print("Using joint limits from URDF file.")
            urdf_bounds = []
            
            # Load the URDF to get joint information
            if self.fk_solver.file_type == ".urdf":
                import urchin
                robot = urchin.URDF.load(self.fk_solver.model_file)
                joint_info = {}
                for joint in robot.joints:
                    joint_info[joint.child] = {
                        'type': joint.joint_type,
                        'axis': joint.axis if hasattr(joint, 'axis') and joint.axis is not None else [0, 0, 1]
                    }
            
            for bone_name in self.controlled_bones:
                if bone_name in self.fk_solver.limits:
                    lower, upper = self.fk_solver.limits[bone_name]
                    #print(f"Bone '{bone_name}' limits: {lower} to {upper}")
                    
                    # Get joint information
                    if bone_name in joint_info:
                        joint_type = joint_info[bone_name]['type']
                        joint_axis = joint_info[bone_name]['axis']
                        
                        if joint_type in ['revolute', 'continuous']:
                            # For revolute joints, apply limits only to the primary rotation axis
                            # Determine which axis has the largest component
                            abs_axis = [abs(x) for x in joint_axis]
                            main_axis = abs_axis.index(max(abs_axis))
                            
                            # Create bounds for X, Y, Z rotations
                            axis_bounds = [(-10, 10), (-10, 10), (-10, 10)]  # Default small range
                            axis_bounds[main_axis] = (lower, upper)  # Apply real limits to main axis
                            
                            urdf_bounds.extend(axis_bounds)
                            #print(f"  Applied limits to axis {main_axis}: {axis_bounds}")
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

    def solve_guess(
        self,
        initial_rotations: np.ndarray,
        learning_rate: float = 0.2,
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
        initial_rotations: np.ndarray = None,
        learning_rate: float = 0.2,
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
    export_file: str = "ik_frames.json"
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


def export_all_frames(
    trajectories: list,
    controlled_bones: list,
    export_file: str = "ik_all_trajectories.json"
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


def compute_objective_breakdown(
    X: np.ndarray,
    objective_list: list,
    fk_solver: "FKSolver"
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
        numeric = float(contribution) if isinstance(contribution, (float, np.number)) else float(contribution.item())
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
    parser.add("--model_file", type=str, default="/home/mei/Downloads/robots/pepper_description-master/urdf/pepper.urdf", help="Path to the GLB, GLTF, or URDF model file.")
    parser.add("--hand", type=str, choices=["left", "right"], default="left", help="For GLTF models, specify hand.")
    parser.add("--bounds", type=str, default=None, help="JSON string for joint bounds, e.g., '[[-10, 10], ...]'")
    parser.add("--controlled_bones", type=str, default="[\"LShoulder\",\"LBicep\",\"LForeArm\",\"l_wrist\"]", help="JSON string of bone names to control.")
    parser.add("--end_effector_bone", type=str, default="LFinger13_link", help="Name of the end-effector bone for the target.")
    parser.add("--threshold", type=float, default=0.005)
    parser.add("--num_steps", type=int, default=10000)
    parser.add("--target_points", type=str, default=None, help="JSON string of target points, e.g., '[[0,0,1], ...]'")
    parser.add("--learning_rate", type=float, default=0.2)
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
            raise ValueError("For URDF files, --controlled_bones and --end_effector_bone must be provided.")
        controlled_bones = json.loads(args.controlled_bones)
        end_effector = args.end_effector_bone
        bounds = json.loads(args.bounds) if args.bounds else None
    else:  # GLTF/GLB
        hand = args.hand
        controlled_bones = [f"{hand}_collar", f"{hand}_shoulder", f"{hand}_elbow", f"{hand}_wrist"]
        end_effector = f"{hand}_index3"
        if args.bounds is None:
            bounds = [(-60, 60)] * 3 * len(controlled_bones) # Default wide bounds
        else:
            bounds = [tuple(b) for b in json.loads(args.bounds)]

    if args.target_points:
        targets = [np.array(p) for p in json.loads(args.target_points)]
    else:
        targets = [np.array([0.3, 0.3, 0.35])] # Default target

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
        print(f"Error: The following bones were not found in the skeleton: {missing_bones}")
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
        mandatory_obj_fns = [DistanceObjTraj(target_points=[target], bone_name=end_effector, use_head=True, weight=1.0)]
        optional_obj_fns = [
            BoneZeroRotationObj(weight=args.additional_objective_weight),
            SDFSelfCollisionPenaltyObj(
                bone_names=controlled_bones,
                num_samples_per_bone=5,
                min_dist=0.02,
                weight=1.0
            )
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
    solver.render(
        angle_vector=final_angles,
        target_pos=targets,
        interactive=True
    )

if __name__ == "__main__":
    main()
