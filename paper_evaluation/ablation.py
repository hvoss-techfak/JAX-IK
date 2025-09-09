import json
import time
from functools import partial

import configargparse
import jax
import jax.numpy as jnp
import numpy as np
from helper import load_skeleton_from_gltf
from tqdm import tqdm
from vedo import Line, Sphere, show


@jax.jit
def jitted_euler_to_matrix(angles):
    # Assumes angles = [angle_x, angle_y, angle_z] and returns R = R_z @ R_y @ R_x.
    cx = jnp.cos(angles[0])
    sx = jnp.sin(angles[0])
    cy = jnp.cos(angles[1])
    sy = jnp.sin(angles[1])
    cz = jnp.cos(angles[2])
    sz = jnp.sin(angles[2])
    R_x = jnp.array([[1, 0, 0, 0],
                     [0, cx, -sx, 0],
                     [0, sx, cx, 0],
                     [0, 0, 0, 1]], dtype=jnp.float32)
    R_y = jnp.array([[cy, 0, sy, 0],
                     [0, 1, 0, 0],
                     [-sy, 0, cy, 0],
                     [0, 0, 0, 1]], dtype=jnp.float32)
    R_z = jnp.array([[cz, -sz, 0, 0],
                     [sz, cz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=jnp.float32)
    return R_z @ R_y @ R_x


@jax.jit
def jitted_look_at_penalty(head, tail, target_point):
    bone_direction = tail - head
    bone_direction = bone_direction / jnp.linalg.norm(bone_direction + 1e-6)
    target_direction = target_point - head
    target_direction = target_direction / jnp.linalg.norm(target_direction + 1e-6)
    cos_theta = jnp.clip(jnp.dot(bone_direction, target_direction), -1.0, 1.0)
    misalignment_angle = jnp.arccos(cos_theta)
    return misalignment_angle ** 2


@jax.jit
def matrix_to_euler(R):
    """
    Convert a 4x4 rotation matrix (assumed to be R = R_z @ R_y @ R_x)
    back into a 3–element Euler angle vector [angle_x, angle_y, angle_z].
    (Note: no full singularity handling is provided.)
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31 = R[2, 0]
    # For our convention, we set:
    angle_y = -jnp.arcsin(jnp.clip(r31, -1.0, 1.0))
    angle_x = jnp.arctan2(R[2, 1], R[2, 2])
    angle_z = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.array([angle_x, angle_y, angle_z], dtype=jnp.float32)


@jax.jit
def rotation_matrix_from_axis_angle(axis, angle):
    """
    Create a 4x4 rotation matrix from a normalized 3D axis and an angle.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    t = 1 - c
    x, y, z = axis[0], axis[1], axis[2]
    R3 = jnp.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
    ], dtype=jnp.float32)
    R4 = jnp.eye(4, dtype=jnp.float32).at[:3, :3].set(R3)
    return R4


@partial(jax.jit, static_argnames=("controlled_indices",))
def _compute_fk(local_array, parent_indices, default_rotations, controlled_indices, angle_vector):
    """
    Pure forward kinematics computation.
    """
    rotations = default_rotations
    # Replace the rotation for each controlled bone using the provided angles.
    for j, bone_idx in enumerate(controlled_indices):
        # Each controlled bone uses 3 angles.
        angles = angle_vector[3 * j: 3 * j + 3]
        rotations = rotations.at[bone_idx].set(jitted_euler_to_matrix(angles))
    n = local_array.shape[0]
    init_carry = jnp.zeros((n, 4, 4), dtype=jnp.float32)

    def fk_scan(carry, inputs):
        i, local_i, rotation_i, parent_idx = inputs
        parent_transform = jax.lax.select(
            parent_idx < 0,
            jnp.eye(4, dtype=jnp.float32),
            carry[parent_idx]
        )
        current = parent_transform @ local_i @ rotation_i
        carry = carry.at[i].set(current)
        return carry, current

    indices = jnp.arange(n)
    inputs = (indices, local_array, rotations, parent_indices)
    final_carry, _ = jax.lax.scan(fk_scan, init_carry, inputs)
    return final_carry

@partial(jax.jit, static_argnames=("fksolver", "hand", "learning_rate"))
def solve_single_ik(target, init_rot, lower_bounds, upper_bounds,
                    fksolver, hand="left", known_rotations=(),
                    threshold=0.01, num_steps=1000, learning_rate=0.1,use_custom_objective=False):
    """
    Solves the IK problem for one target point with a specified learning rate.

    Parameters:
       target: 3-element target point.
       init_rot: Initial angle vector.
       lower_bounds, upper_bounds: Bound arrays.
       fksolver: An instance of FKSolver.
       hand: "left" or "right".
       known_rotations: Optional known rotations.
       threshold: Objective threshold for convergence.
       num_steps: Maximum number of iterations.
       learning_rate: The gradient descent step size.

    Returns:
       best_angles: The optimized angles.
       best_obj: The final objective value.
       step: The number of iterations taken.
    """
    target = jnp.array(target, dtype=jnp.float32)
    print("using init_rot",init_rot)
    init_rot = jnp.array(init_rot, dtype=jnp.float32).flatten()

    ik_problem = CustomIKProblem(
        fksolver, target, lower_bounds, upper_bounds,
        penalty_weight=0.25, configuration="hand_down", hand=hand,
        known_rotations=known_rotations
    )

    def body_fn(state):
        i, angles, best_angles, best_obj = state
        grad = jax.grad(ik_problem._objective_jax_pure)(angles)
        new_angles = jnp.clip(angles - learning_rate * grad, lower_bounds, upper_bounds)
        new_obj = ik_problem._objective_jax_pure(new_angles)
        new_best_angles = jnp.where(new_obj < best_obj, new_angles, best_angles)
        new_best_obj = jnp.minimum(new_obj, best_obj)
        return (i + 1, new_angles, new_best_angles, new_best_obj)

    def cond_fn(state):
        i, angles, best_angles, best_obj = state
        return jnp.logical_and(i < num_steps, best_obj > threshold)

    init_obj = ik_problem._objective_jax_pure(init_rot)
    init_state = (0, init_rot, init_rot, init_obj)
    step, _, best_angles, best_obj = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return best_angles, best_obj, step


@partial(jax.jit, static_argnames=("fk_solver", "controlled_indices", "max_iterations", "use_custom_objective"))
def solve_single_ik_ccd(target, init_angles, lower_bounds, upper_bounds,
                        fk_solver, controlled_indices, controlled_map_array,
                        effector_index, max_iterations, epsilon, use_custom_objective=False):
    """
    CCD solver that iteratively “sweeps” through the chain from the end effector back to the base.

    With use_custom_objective=True the solver uses a custom objective (via a CustomIKProblem instance)
    not only for the stop condition but also to decide whether to accept each bone update.

    Parameters:
       target: (3,) target position (jnp.array).
       init_angles: initial angle vector (flattened, one entry per controlled DOF).
       lower_bounds, upper_bounds: per–angle bounds.
       fk_solver: the FK solver instance.
       controlled_indices: tuple of bone indices (in chain order, base→tip).
       controlled_map_array: a jnp.array mapping full bone index -> angle vector slice start.
       effector_index: full–skeleton index of the target (end–effector) bone.
       max_iterations: maximum number of outer iterations.
       epsilon: convergence threshold (if not using a custom objective, this is the end effector error).
       use_custom_objective: if True then use the custom objective both as stop condition and as the optimization metric.

    Returns:
       (final_angles, final_objective (or error), iterations)
    """
    num_controlled = len(controlled_indices)
    # Precompute the reversed controlled chain (for CCD, update from tip to base)
    controlled_rev = jnp.array(tuple(reversed(controlled_indices)), dtype=jnp.int32)

    # When using a custom objective, create a CustomIKProblem instance.
    if use_custom_objective:
        ik_problem = CustomIKProblem(
            fk_solver, target, lower_bounds, upper_bounds,
            penalty_weight=0.25, configuration="hand_down", hand="left",  # note: `hand` must be defined in your context
            known_rotations=None
        )

        def ccd_cond(state):
            it, angles = state
            obj = ik_problem._objective_jax_pure(angles)
            return jnp.logical_and(obj > epsilon, it < max_iterations)
    else:
        def ccd_cond(state):
            it, angles = state
            fk = fk_solver.compute_fk_from_angles(angles)
            head_eff = fk[effector_index][:3, 3]
            err = jnp.linalg.norm(head_eff - target)
            return jnp.logical_and(err > epsilon, it < max_iterations)

    def ccd_body(state):
        it, angles = state
        # Recompute FK so that each update uses the latest chain.
        fk = fk_solver.compute_fk_from_angles(angles)

        # (For CCD, each bone is updated in reverse order.)
        def update_bone(j, angles_in):
            fk = fk_solver.compute_fk_from_angles(angles_in)
            head_eff = fk[effector_index][:3, 3]
            bone_idx = controlled_rev[j]
            pivot = fk[bone_idx][:3, 3]
            curr_vec = head_eff - pivot
            tgt_vec = target - pivot
            norm_curr = jnp.linalg.norm(curr_vec) + 1e-6
            norm_tgt = jnp.linalg.norm(tgt_vec) + 1e-6
            curr_dir = curr_vec / norm_curr
            tgt_dir = tgt_vec / norm_tgt

            dot_val = jnp.clip(jnp.dot(curr_dir, tgt_dir), -1.0, 1.0)
            axis = jnp.cross(curr_dir, tgt_dir)
            norm_axis = jnp.linalg.norm(axis) + 1e-6
            angle_delta = jnp.arctan2(norm_axis, dot_val)
            axis_normalized = axis / norm_axis

            R_delta = rotation_matrix_from_axis_angle(axis_normalized, angle_delta)
            parent_idx = fk_solver.parent_indices[bone_idx]
            parent_transform = jax.lax.select(
                parent_idx < 0,
                jnp.eye(4, dtype=jnp.float32),
                fk[parent_idx]
            )
            parent_rot = parent_transform[:3, :3]
            angle_start = controlled_map_array[bone_idx]
            current_angles = jax.lax.dynamic_slice(angles_in, (angle_start,), (3,))
            R_current_local = jitted_euler_to_matrix(current_angles)[:3, :3]
            new_R_local = parent_rot.T @ R_delta[:3, :3] @ parent_rot @ R_current_local
            new_R_local_4x4 = jnp.eye(4, dtype=jnp.float32).at[:3, :3].set(new_R_local)
            new_angles_local = matrix_to_euler(new_R_local_4x4)
            # Clamp the per–bone angles.
            lb = jax.lax.dynamic_slice(lower_bounds, (angle_start,), (3,))
            ub = jax.lax.dynamic_slice(upper_bounds, (angle_start,), (3,))
            new_angles_local = jnp.clip(new_angles_local, lb, ub)
            candidate_angles = jax.lax.dynamic_update_slice(angles_in, new_angles_local, (angle_start,))
            candidate_angles = jnp.clip(candidate_angles, lower_bounds, upper_bounds)
            # If using the custom objective, accept the update only if it lowers the objective.
            if use_custom_objective:
                current_obj = ik_problem._objective_jax_pure(angles_in)
                candidate_obj = ik_problem._objective_jax_pure(candidate_angles)
                new_angles_final = jax.lax.cond(candidate_obj < current_obj,
                                                lambda _: candidate_angles,
                                                lambda _: angles_in,
                                                operand=None)
                return new_angles_final
            else:
                return candidate_angles

        angles_new = jax.lax.fori_loop(0, num_controlled, update_bone, angles)
        return (it + 1, angles_new)

    init_state = (0, init_angles)
    it_final, final_angles = jax.lax.while_loop(ccd_cond, ccd_body, init_state)
    # Compute the final objective.
    if use_custom_objective:
        final_obj = ik_problem._objective_jax_pure(final_angles)
    else:
        fk_final = fk_solver.compute_fk_from_angles(final_angles)
        head_eff_final = fk_final[effector_index][:3, 3]
        final_obj = jnp.linalg.norm(head_eff_final - target)
    return final_angles, final_obj, it_final


@partial(jax.jit, static_argnames=("fk_solver", "controlled_indices", "max_iterations", "use_custom_objective"))
def solve_single_ik_fabrik(target, init_angles, lower_bounds, upper_bounds,
                           fk_solver, controlled_indices, controlled_map_array,
                           effector_index, max_iterations, epsilon, use_custom_objective=False):
    """
    FABRIK solver that works on the joint positions of the IK chain.

    With use_custom_objective=True the custom objective (via a CustomIKProblem instance) is used
    both for the stop condition and to decide whether to accept each bone update.

    Parameters:
       target: (3,) target position.
       init_angles: initial angle vector.
       lower_bounds, upper_bounds: per–angle bounds.
       fk_solver: the FK solver instance.
       controlled_indices: tuple of bone indices (in chain order).
       controlled_map_array: maps bone index -> start index in the angle vector.
       effector_index: index of the end–effector bone.
       max_iterations: maximum iterations.
       epsilon: convergence threshold (if not using a custom objective, this is the end effector error).
       use_custom_objective: if True then use the custom objective as the optimization metric.

    Returns:
       (final_angles, final_objective (or error), iterations)
    """
    static_controlled = jnp.array(controlled_indices, dtype=jnp.int32)
    num_controlled = len(controlled_indices)
    n_joints = num_controlled + 1  # controlled joints plus end effector

    # Compute constant bone lengths and fixed root from the initial configuration.
    fk_init = fk_solver.compute_fk_from_angles(init_angles)

    def get_joint(idx):
        return fk_init[idx][:3, 3]

    joints_init = jnp.stack(
        [get_joint(bone_idx) for bone_idx in controlled_indices] +
        [fk_init[effector_index][:3, 3]],
        axis=0)
    bone_lengths_const = jnp.linalg.norm(joints_init[1:] - joints_init[:-1], axis=1)
    root_const = joints_init[0]
    total_length = jnp.sum(bone_lengths_const)
    # Precompute the default (rest) directions for each bone.
    default_dirs = joints_init[1:] - joints_init[:-1]
    default_dirs = default_dirs / (jnp.linalg.norm(default_dirs, axis=1, keepdims=True) + 1e-6)

    if use_custom_objective:
        ik_problem = CustomIKProblem(
            fk_solver, target, lower_bounds, upper_bounds,
            penalty_weight=0.25, configuration="hand_down", hand="left",
            known_rotations=None
        )

        def fabrik_cond(state):
            it, angles = state
            obj = ik_problem._objective_jax_pure(angles)
            return jnp.logical_and(obj > epsilon, it < max_iterations)
    else:
        def fabrik_cond(state):
            it, angles = state
            fk = fk_solver.compute_fk_from_angles(angles)
            eff_pos = fk[effector_index][:3, 3]
            err = jnp.linalg.norm(eff_pos - target)
            return jnp.logical_and(err > epsilon, it < max_iterations)

    def fabrik_body(state):
        it, angles = state
        fk = fk_solver.compute_fk_from_angles(angles)
        # Compute current joint positions from FK (but fix the root).
        joints_fk = [fk[idx][:3, 3] for idx in controlled_indices]
        joints_fk.append(fk[effector_index][:3, 3])
        joints = jnp.stack(joints_fk, axis=0)
        joints = joints.at[0].set(root_const)
        target_dist = jnp.linalg.norm(target - root_const)

        # If target is unreachable, stretch the chain.
        def unreachable(joints):
            def body_unreachable(i, joints_in):
                prev = joints_in[i]
                dir_vec = (target - prev) / (jnp.linalg.norm(target - prev) + 1e-6)
                return joints_in.at[i + 1].set(prev + dir_vec * bone_lengths_const[i])

            return jax.lax.fori_loop(0, n_joints - 1, body_unreachable, joints)

        joints = jax.lax.cond(target_dist > total_length, unreachable, lambda x: x, joints)

        # Backward pass: set effector to target and update backward.
        joints = joints.at[-1].set(target)

        def backward_body(i, joints_in):
            j = n_joints - 2 - i
            dir_vec = (joints_in[j] - joints_in[j + 1]) / (jnp.linalg.norm(joints_in[j] - joints_in[j + 1]) + 1e-6)
            return joints_in.at[j].set(joints_in[j + 1] + dir_vec * bone_lengths_const[j])

        joints = jax.lax.fori_loop(0, n_joints - 1, backward_body, joints)

        # Forward pass: reset root and update forward.
        joints = joints.at[0].set(root_const)

        def forward_body(i, joints_in):
            j = i
            dir_vec = (joints_in[j + 1] - joints_in[j]) / (jnp.linalg.norm(joints_in[j + 1] - joints_in[j]) + 1e-6)
            return joints_in.at[j + 1].set(joints_in[j] + dir_vec * bone_lengths_const[j])

        joints = jax.lax.fori_loop(0, n_joints - 1, forward_body, joints)

        # Now update each controlled bone’s rotation.
        def update_bone(i, angles_in):
            bone_idx = static_controlled[i]
            desired_vec = joints[i + 1] - joints[i]
            desired_norm = desired_vec / (jnp.linalg.norm(desired_vec) + 1e-6)
            default_dir = default_dirs[i]
            dot_val = jnp.clip(jnp.dot(default_dir, desired_norm), -1.0, 1.0)
            angle_needed = jnp.arccos(dot_val)
            axis = jnp.cross(default_dir, desired_norm)
            norm_axis = jnp.linalg.norm(axis) + 1e-6
            axis_normalized = axis / norm_axis
            R_global = rotation_matrix_from_axis_angle(axis_normalized, angle_needed)
            parent_idx = fk_solver.parent_indices[bone_idx]
            parent_transform = jax.lax.select(
                parent_idx < 0,
                jnp.eye(4, dtype=jnp.float32),
                fk[parent_idx]
            )
            parent_rot = parent_transform[:3, :3]
            new_R_local = parent_rot.T @ R_global[:3, :3]
            new_R_local_4x4 = jnp.eye(4, dtype=jnp.float32).at[:3, :3].set(new_R_local)
            new_angles_local = matrix_to_euler(new_R_local_4x4)
            angle_start = controlled_map_array[bone_idx]
            lb = jax.lax.dynamic_slice(lower_bounds, (angle_start,), (3,))
            ub = jax.lax.dynamic_slice(upper_bounds, (angle_start,), (3,))
            new_angles_local = jnp.clip(new_angles_local, lb, ub)
            candidate_angles = jax.lax.dynamic_update_slice(angles_in, new_angles_local, (angle_start,))
            candidate_angles = jnp.clip(candidate_angles, lower_bounds, upper_bounds)
            if use_custom_objective:
                current_obj = ik_problem._objective_jax_pure(angles_in)
                candidate_obj = ik_problem._objective_jax_pure(candidate_angles)
                new_angles_final = jax.lax.cond(candidate_obj < current_obj,
                                                lambda _: candidate_angles,
                                                lambda _: angles_in,
                                                operand=None)
                return new_angles_final
            else:
                return candidate_angles

        angles_new = jax.lax.fori_loop(0, num_controlled, update_bone, angles)
        return (it + 1, angles_new)

    init_state = (0, init_angles)
    it_final, final_angles = jax.lax.while_loop(fabrik_cond, fabrik_body, init_state)
    if use_custom_objective:
        final_obj = ik_problem._objective_jax_pure(final_angles)
    else:
        fk_final = fk_solver.compute_fk_from_angles(final_angles)
        eff_pos_final = fk_final[effector_index][:3, 3]
        final_obj = jnp.linalg.norm(eff_pos_final - target)
    return final_angles, final_obj, it_final


class FKSolver:
    """
    Forward kinematics solver.
    """

    def __init__(self, gltf_file, hand="left", controlled_bones=None):
        self.skeleton = load_skeleton_from_gltf(gltf_file)
        self._prepare_fk_arrays()
        self.hand = hand
        # Use custom controlled bones if provided; otherwise, use defaults.
        if controlled_bones is None:
            controlled_bones = [f"{hand}_collar",
                                f"{hand}_shoulder",
                                f"{hand}_elbow",
                                f"{hand}_wrist"]
        self.controlled_bones = controlled_bones
        # Save controlled indices as a tuple (so they can be static in JIT).
        self.controlled_indices = tuple(
            i for i, name in enumerate(self.bone_names) if name in self.controlled_bones
        )
        self.default_rotations = jnp.stack(
            [jnp.eye(4, dtype=jnp.float32) for _ in self.bone_names],
            axis=0
        )
        # Create a mapping array: for each bone index, store the start index in the angle vector if controlled, else -1.
        controlled_map = -np.ones(len(self.bone_names), dtype=np.int32)
        for j, bone_idx in enumerate(self.controlled_indices):
            controlled_map[bone_idx] = 3 * j
        self.controlled_map_array = jnp.array(controlled_map, dtype=jnp.int32)
        # Determine the effector bone index (e.g., fingertip).
        try:
            self.effector_index = self.bone_names.index(f"{hand}_index3_look")
        except ValueError:
            raise ValueError(f"Effector bone '{hand}_index3_look' not found in skeleton.")

    def _prepare_fk_arrays(self):
        self.bone_names = []
        self.local_list = []
        self.parent_list = []

        def dfs(bone_name, parent_index):
            current_index = len(self.bone_names)
            self.bone_names.append(bone_name)
            bone = self.skeleton[bone_name]
            self.local_list.append(jnp.array(bone["local_transform"], dtype=jnp.float32))
            self.parent_list.append(parent_index)
            for child in bone["children"]:
                dfs(child, current_index)

        roots = [bone["name"] for bone in self.skeleton.values() if bone["parent"] is None]
        for root in roots:
            dfs(root, -1)
        self.local_array = jnp.stack(self.local_list, axis=0)
        self.parent_indices = jnp.array(self.parent_list, dtype=jnp.int32)

    def compute_fk_from_angles(self, angle_vector):
        """
        Compute global transforms from a given angle vector.
        """
        return _compute_fk(
            self.local_array,
            self.parent_indices,
            self.default_rotations,
            self.controlled_indices,
            angle_vector
        )

    def get_bone_head_tail_from_fk(self, fk_transforms, bone_name):
        """
        Return the (head, tail) positions for a given bone.
        """
        try:
            idx = self.bone_names.index(bone_name)
        except ValueError:
            raise ValueError(f"Bone '{bone_name}' not found in skeleton.")
        global_transform = fk_transforms[idx]
        head = global_transform[:3, 3]
        bone = self.skeleton[bone_name]
        tail_local = jnp.array([0, bone["bone_length"], 0, 1], dtype=jnp.float32)
        tail = (global_transform @ tail_local)[:3]
        return head, tail

    def render(self, angle_vector=None, target_pos=[], interactive=False):
        """
        Render the skeleton (and optional target markers) using vedo.
        """
        if angle_vector is None:
            n_angles = len(self.controlled_indices) * 3
            angle_vector = jnp.zeros(n_angles, dtype=jnp.float32)
        if not isinstance(angle_vector, jnp.ndarray):
            angle_vector = jnp.array(angle_vector, dtype=jnp.float32)
        fk_transforms = self.compute_fk_from_angles(angle_vector)
        actors = []
        for bone_name in self.bone_names:
            head, tail = self.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            head_np = np.array(jax.device_get(head))
            tail_np = np.array(jax.device_get(tail))
            actors.append(Line(head_np, tail_np, lw=3, c="blue"))
            actors.append(Sphere(pos=head_np, r=0.02, c="red"))
        for target, color in target_pos:
            actors.append(Sphere(pos=target, r=0.02 if color == "green" else 0.01, c=color))
        show(actors, "Skeleton FK", axes=1, interactive=interactive)


class CustomIKProblem:
    """
    Defines the inverse kinematics objective.
    """

    def __init__(self, fksolver, target_pos, lower_bounds, upper_bounds,
                 penalty_weight=1.0, known_rotations=(), hand="left",
                 configuration="pointing"):
        self.fksolver = fksolver  # An instance of FKSolver.
        self.lower_bounds = jnp.array(lower_bounds, dtype=jnp.float32)
        self.upper_bounds = jnp.array(upper_bounds, dtype=jnp.float32)
        self.target_pos = jnp.array(target_pos, dtype=jnp.float32)
        self.penalty_weight = penalty_weight
        self.hand = hand
        self.known_rotations = known_rotations  # For an additional difference penalty.
        self.configuration = configuration

    def _objective_jax_pure(self, angle_vector):
        """
        Pure objective function for IK.
        """
        angle_vector = jnp.clip(angle_vector, self.lower_bounds, self.upper_bounds)
        fk_transforms = self.fksolver.compute_fk_from_angles(angle_vector)

        # Get the fingertip position.
        head_index, tail_index = self.fksolver.get_bone_head_tail_from_fk(
            fk_transforms, f"{self.hand}_index3_look")
        fingertip = head_index
        distance = jnp.linalg.norm(fingertip - self.target_pos)

        # Get additional look–at targets.
        head_wrist, tail_wrist = self.fksolver.get_bone_head_tail_from_fk(
            fk_transforms, f"{self.hand}_wrist")
        head_shoulder, tail_shoulder = self.fksolver.get_bone_head_tail_from_fk(
            fk_transforms, f"{self.hand}_shoulder")
        target_wrist_look = tail_wrist
        target_index3_look = tail_index
        target_shoulder_look = tail_shoulder
        target_shoulder_look = target_shoulder_look.at[2].set(target_shoulder_look[2] - 1.0)
        target_wrist_look = target_wrist_look.at[1].set(target_wrist_look[1] + 1.0)
        target_index3_look = target_index3_look.at[2].set(target_index3_look[2] + 1.0)

        wrist_penalty = jitted_look_at_penalty(head_wrist, tail_wrist, target_wrist_look)
        index_penalty = jitted_look_at_penalty(head_index, tail_index, target_index3_look)
        shoulder_penalty = jitted_look_at_penalty(head_shoulder, tail_shoulder, target_shoulder_look) * 0.002

        if isinstance(self.known_rotations, tuple) and len(self.known_rotations) == 2:
            candidate_known, mask = self.known_rotations
            diff_sq = jnp.sum((angle_vector - candidate_known) ** 2, axis=1)
            diff_sq = diff_sq * mask.astype(jnp.float32)
            valid_count = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
            difference_penalty = (jnp.sum(diff_sq) / valid_count) / 20.0
        elif isinstance(self.known_rotations, (list, tuple)) and len(self.known_rotations) > 0:
            diff_penalties = [jnp.sum((angle_vector - init) ** 2)
                              for init in self.known_rotations]
            difference_penalty = jnp.mean(jnp.array(diff_penalties)) / 20.0
        else:
            difference_penalty = 0.0

        total_penalty = (wrist_penalty + index_penalty + shoulder_penalty) / 3.0
        total_objective = distance + (total_penalty + difference_penalty) * self.penalty_weight
        return total_objective


class InverseKinematicsSolver:
    """
    Library-style Inverse Kinematics (IK) solver.
    """

    def __init__(self, gltf_file, hand="left", controlled_bones=None, bounds=None,
                 penalty_weight=0.25, threshold=0.01, num_steps=1000, optimize_jax_cache=True):
        """
        Parameters:
            gltf_file (str): Path to the GLTF file.
            hand (str): "left" or "right" (used for default naming).
            controlled_bones (list of str, optional): List of bone names to control.
            bounds (list of tuple, optional): A list of (lower, upper) tuples (in degrees)
                for each angle for each controlled bone.
            penalty_weight (float): Weight for the penalty term.
            threshold (float): Optimization stops when the objective falls below this value.
            num_steps (int): Maximum number of optimization iterations.
        """
        self.hand = hand
        self.fk_solver = FKSolver(gltf_file, hand, controlled_bones)
        self.controlled_bones = self.fk_solver.controlled_bones

        if optimize_jax_cache:
            jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
            jax.config.update('jax_default_device', jax.devices('cpu')[0])

        # Set default bounds if not provided.
        if bounds is None:
            if hand == "left":
                bounds = [
                    (-10, 10), (-10, 10), (-10, 10),
                    (0, 50), (-140, 50), (-70, 25),
                    (-45, 90), (-180, 5), (-10, 10),
                    (-90, 90), (-70, 70), (-55, 85)
                ]
            else:
                bounds = [
                    (-10, 10), (-10, 10), (-10, 10),
                    (-50, 0), (-50, 140), (-25, 70),
                    (-90, 45), (-5, 180), (-10, 10),
                    (-90, 90), (-70, 70), (-55, 85)
                ]
        # Convert bounds from degrees to radians.
        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = jnp.array(lower_bounds, dtype=jnp.float32)
        self.upper_bounds = jnp.array(upper_bounds, dtype=jnp.float32)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps

        # Maintain an average iteration time (in seconds) to help set a maximum time budget.
        self.avg_iter_time = None  # initial guess (adjust as needed)

    def solve(self, target_point, initial_rotations=None, known_rotations=(),
              learning_rate=0.2, max_time=None, verbose=False,max_iterations=None,use_custom_objective=False):
        """
        Solve the inverse kinematics problem for a given target point.

        In addition to the usual parameters, if max_time (in seconds) is provided,
        this function will dynamically adjust the maximum allowed number of iterations
        based on the estimated iteration time.

        Parameters:
            target_point (array-like): A 3-element array specifying the target position.
            initial_rotations (array-like, optional): Initial angles (flattened 1D array).
                If None, a zero vector is used.
            known_rotations (optional): Additional rotations used in a difference penalty.
            learning_rate (float): The learning rate to use in the optimization.
            max_time (float, optional): Maximum allowed wall–clock time (in seconds)
                for the solve call.

        Returns:
            tuple: (best_angles, best_objective, steps)
        """
        target_point = np.asarray(target_point)
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        else:
            initial_rotations = np.array(initial_rotations, dtype=np.float32)

        if max_time is not None and self.avg_iter_time:
            allowed_steps = int(max_time / self.avg_iter_time)
            allowed_steps = max(allowed_steps, 1)
            allowed_steps = min(allowed_steps, self.num_steps)
        else:
            allowed_steps = self.num_steps

        if max_iterations is not None:
            allowed_steps = max_iterations

        if verbose and max_time is not None:
            print(f"User set max time. Based on previous step times we dynamically set the allowed steps to: {allowed_steps}")


        start_time = time.time()
        best_angles, best_obj, steps = solve_single_ik(
            target_point,
            initial_rotations,
            self.lower_bounds,
            self.upper_bounds,
            self.fk_solver,
            hand=self.hand,
            known_rotations=known_rotations,
            threshold=self.threshold,
            num_steps=allowed_steps,
            learning_rate=learning_rate,
            use_custom_objective=use_custom_objective
        )
        elapsed_time = time.time() - start_time
        if steps > 0:
            new_iter_time = elapsed_time / steps
            self.avg_iter_time = new_iter_time if self.avg_iter_time is None else 0.9 * self.avg_iter_time + 0.1 * new_iter_time

        if max_time is not None and elapsed_time > max_time and verbose:
            print(f"Time limit exceeded: {elapsed_time:.4f} seconds (allowed {max_time} seconds)")

        if best_obj > self.threshold and verbose:
            print(f"Warning: Optimization did not converge below threshold ({self.threshold}).")

        return np.array(best_angles), float(best_obj), int(steps)

    def solveCCD(self, target_point, initial_rotations=None, max_iterations=100, epsilon=0.005,use_custom_objective=False):
        """
        Solve the IK problem using the CCD algorithm.
        """
        target_point = np.asarray(target_point)
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        else:
            initial_rotations = np.array(initial_rotations, dtype=np.float32)
        angles, err, iters = solve_single_ik_ccd(
            target=jnp.array(target_point, dtype=jnp.float32),
            init_angles=jnp.array(initial_rotations, dtype=jnp.float32).flatten(),
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            fk_solver=self.fk_solver,
            controlled_indices=self.fk_solver.controlled_indices,
            controlled_map_array=self.fk_solver.controlled_map_array,
            effector_index=self.fk_solver.effector_index,
            max_iterations=max_iterations,
            epsilon=epsilon,
            use_custom_objective=use_custom_objective
        )
        return np.array(angles), float(err), int(iters)

    def solveFABRIK(self, target_point, initial_rotations=None, max_iterations=100, epsilon=0.005,use_custom_objective=False):
        """
        Solve the IK problem using the FABRIK algorithm.
        """
        target_point = np.asarray(target_point)
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        else:
            initial_rotations = np.array(initial_rotations, dtype=np.float32)

        angles, err, iters = solve_single_ik_fabrik(
            target=jnp.array(target_point, dtype=jnp.float32),
            init_angles=jnp.array(initial_rotations, dtype=jnp.float32).flatten(),
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            fk_solver=self.fk_solver,
            controlled_indices=self.fk_solver.controlled_indices,
            controlled_map_array=self.fk_solver.controlled_map_array,
            effector_index=self.fk_solver.effector_index,
            max_iterations=max_iterations,
            epsilon=epsilon,
            use_custom_objective=use_custom_objective
        )
        return np.array(angles), float(err), int(iters)

    def render(self, angle_vector, target_pos=[], interactive=False):
        """
        Render the skeleton with the given pose.
        """
        self.fk_solver.render(angle_vector=angle_vector, target_pos=target_pos, interactive=interactive)



def main():
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"]
    )
    parser.add("--gltf_file", type=str, default="../smplx.glb", help="Path to the glTF file.")
    parser.add("--hand", type=str, choices=["left", "right"], default="left", help="Which hand to use.")
    parser.add("--bounds", type=str, default=None, help="List of bounds as a JSON string.")
    parser.add("--controlled_bones", type=str, default=None, help="Comma-separated list of controlled bones.")
    parser.add("--threshold", type=float, default=0.005, help="Threshold value for the solver.")
    parser.add("--num_steps", type=int, default=500, help="Number of steps for the solver.")
    parser.add("--target_points", type=str, default=None, help="List of target points as a JSON string.")
    parser.add("--learning_rate", type=float, default=0.1, help="Learning rate for the solver.")
    parser.add("--max_iterations", type=int, default=500, help="Maximum iterations for each solver call.")
    parser.add("--solver_type", type=str, choices=["gd", "ccd", "fabrik"], default="gd",
               help="Select the solver: gd (Gradient Descent), ccd, or fabrik.")
    parser.add("--custom_objective", type=bool, default=False, help="Use the custom objective for the solver.")

    args = parser.parse_args()

    gltf_file = args.gltf_file
    hand = args.hand

    bounds = json.loads(args.bounds) if args.bounds else None
    controlled_bones = args.controlled_bones.split(",") if args.controlled_bones else None

    target_list = json.loads(args.target_points) if args.target_points else [[0.15, 0, 0.35]]
    target = [np.array(pt) for pt in target_list]

    solver = InverseKinematicsSolver(
        gltf_file, controlled_bones=controlled_bones, bounds=bounds, threshold=args.threshold, num_steps=args.num_steps,
        hand=hand
    )


    tbar = tqdm(range(len(target)))
    avg_steps = 0

    solved_success = 0

    time1 = time.time()

    for i in tbar:
        time_iter = time.time()
        if args.solver_type == "gd":
            best_angles, obj, steps = solver.solve(
                target_point=target[i], learning_rate=args.learning_rate, max_iterations=args.max_iterations,use_custom_objective=args.custom_objective
            )
        elif args.solver_type == "ccd":
            best_angles, obj, steps = solver.solveCCD(
                target_point=target[i], max_iterations=args.max_iterations,use_custom_objective=args.custom_objective
            )
        elif args.solver_type == "fabrik":
            best_angles, obj, steps = solver.solveFABRIK(
                target_point=target[i], max_iterations=args.max_iterations,use_custom_objective=args.custom_objective
            )
        if obj < args.threshold:
            solved_success += 1

        if i > 10:
            print(f"Time for iteration {i}: {time.time() - time_iter:.4f} seconds. Steps: {steps}. Success: {obj < args.threshold}")

        if i == 10:
            time1 = time.time()
            avg_steps = 0

        avg_steps += steps
        tbar.set_description(f"Error: {obj:.4f} after {steps} iterations")
    solved_success_rate = solved_success / len(target)

    print(f"Result: error = {obj}, iterations = {steps}, average = {avg_steps}")
    print(f"Solving {len(target)} times with total {avg_steps} iterations took {time.time() - time1:.4f} seconds")
    print(f"Success rate: {solved_success_rate:.4f}")


if __name__ == "__main__":
    main()
