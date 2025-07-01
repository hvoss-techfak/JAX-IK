import json
import time

import configargparse
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm
from vedo import Line, Sphere, show

from helper import load_skeleton_from_gltf


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
    bone_direction = bone_direction / (jnp.linalg.norm(bone_direction) + 1e-6)
    target_direction = target_point - head
    target_direction = target_direction / (jnp.linalg.norm(target_direction) + 1e-6)
    cos_theta = jnp.clip(jnp.dot(bone_direction, target_direction), -1.0, 1.0)
    misalignment_angle = jnp.arccos(cos_theta)
    return misalignment_angle ** 2


@jax.jit
def matrix_to_euler(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31 = R[2, 0]
    angle_y = -jnp.arcsin(jnp.clip(r31, -1.0, 1.0))
    angle_x = jnp.arctan2(R[2, 1], R[2, 2])
    angle_z = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.array([angle_x, angle_y, angle_z], dtype=jnp.float32)


@jax.jit
def rotation_matrix_from_axis_angle(axis, angle):
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
    rotations = default_rotations
    # Replace the rotation for each controlled bone using the provided angles.
    for j, bone_idx in enumerate(controlled_indices):
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


def distance_obj_traj(bone_name, use_head=False, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        target_bone = head if use_head else tail
        return jnp.linalg.norm(target_bone - target_point) * weight
    return obj

def look_at_obj_traj(bone_name, use_head, modifications, weight):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        adjusted_target = head if use_head else tail
        for idx, delta in modifications:
            adjusted_target = adjusted_target.at[idx].set(adjusted_target[idx] + delta)
        penalty = jitted_look_at_penalty(head, tail, adjusted_target)
        return weight * penalty
    return obj

def known_rot_obj_traj(candidate_known, mask, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        angles_reshaped = config.reshape(-1, 3)
        candidate = jnp.array(candidate_known, dtype=jnp.float32)
        mask_arr = jnp.array(mask, dtype=jnp.float32)
        diff_sq = jnp.sum((angles_reshaped - candidate) ** 2, axis=1)
        diff_sq = diff_sq * mask_arr
        valid_count = jnp.maximum(jnp.sum(mask_arr), 1.0)
        return jnp.sum(diff_sq) / valid_count * weight
    return obj

def collision_penalty_obj_traj(collider_eqs, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        joint_positions = []
        for bone_name in fksolver.bone_names:
            head, _ = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            joint_positions.append(head)
        joint_positions = jnp.stack(joint_positions, axis=0)
        total_penalty = 0.0
        for eq in collider_eqs:
            eq_jax = jnp.array(eq, dtype=jnp.float32)
            def point_penalty(p):
                d = jnp.max(eq_jax[:, :3] @ p + eq_jax[:, 3])
                return jnp.square(jnp.maximum(0.0, -d))
            penalties = jax.vmap(point_penalty)(joint_positions)
            total_penalty += jnp.sum(penalties)
        return total_penalty * weight
    return obj

def velocity_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1:
            return 0.0
        vel = X[1:] - X[:-1]
        return weight * jnp.sum(jnp.square(vel))
    return obj

def acceleration_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1 or X.shape[0] < 3:
            return 0.0
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        return weight * jnp.sum(jnp.square(acc))
    return obj

def jerk_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1 or X.shape[0] < 4:
            return 0.0
        jerk = X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]
        return weight * jnp.sum(jnp.square(jerk))
    return obj

def init_pose_obj(init_rot, weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1:
            return 0.0
        return weight * jnp.sum(jnp.square(X[0] - init_rot))
    return obj


@partial(jax.jit, static_argnames=("fksolver", "objective_functions", "T"))
def solve_ik(target_point, init_rot, lower_bounds, upper_bounds, objective_functions,
             fksolver, T, threshold=0.01, num_steps=1000, learning_rate=0.1,
             beta1=0.9, beta2=0.999, epsilon=1e-8):
    D = init_rot.shape[0]
    if T == 1:
        # For a single–configuration case, our variable is 1D.
        x0 = init_rot
        lower_bounds_traj = lower_bounds
        upper_bounds_traj = upper_bounds
    else:
        # For a trajectory, we use a flattened variable of shape (T*D,)
        x0 = jnp.tile(init_rot, (T,))
        lower_bounds_traj = jnp.tile(lower_bounds, T)
        upper_bounds_traj = jnp.tile(upper_bounds, T)

    def total_objective(x_flat):
        if T == 1:
            X = x_flat  # shape (D,)
        else:
            X = x_flat.reshape((T, D))
        total = 0.0
        for fn in objective_functions:
            total += fn(X, fksolver, target_point)
        return total

    objective_fn = total_objective
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    i0 = 0
    init_obj_val = objective_fn(x0)
    state = (i0, x0, m0, v0, x0, init_obj_val)

    def body_fn(state):
        i, x, m, v, best_x, best_obj = state
        # Compute the gradient of the objective with respect to x.
        grad_val = jax.grad(objective_fn)(x)

        # Update the exponential moving averages.
        m_new = beta1 * m + (1 - beta1) * grad_val
        v_new = beta2 * v + (1 - beta2) * (grad_val ** 2)

        # Compute bias corrections as in the PyTorch implementation.
        bias_correction1 = 1 - beta1 ** (i + 1)
        bias_correction2 = 1 - beta2 ** (i + 1)
        step_size = learning_rate * jnp.sqrt(bias_correction2) / bias_correction1
        denom = jnp.sqrt(v_new) + epsilon

        # Apply the masking trick: compute a mask where m_new and grad have the same sign.
        mask = jnp.where(m_new * grad_val > 0, 1.0, 0.0)
        # Normalize the mask by dividing by its mean (clamped to a minimum to avoid division by near-zero).
        mask = mask / jnp.maximum(jnp.mean(mask), 1e-3)

        # Adapted from https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py
        # Use the masked m_new to compute a "normalized gradient".
        norm_grad = (m_new * mask) / denom
        new_x = x - step_size * norm_grad
        new_x = jnp.clip(new_x, lower_bounds_traj, upper_bounds_traj)

        new_obj = objective_fn(new_x)
        new_best_x = jnp.where(new_obj < best_obj, new_x, best_x)
        new_best_obj = jnp.minimum(new_obj, best_obj)

        return (i + 1, new_x, m_new, v_new, new_best_x, new_best_obj)

    def cond_fn(state):
        i, x, m, v, best_x, best_obj = state
        return jnp.logical_and(i < num_steps, best_obj > threshold)

    final_state = jax.lax.while_loop(cond_fn, body_fn, state)
    i_final, final_x, m_final, v_final, best_x, best_obj = final_state
    if T == 1:
        return best_x, best_obj, i_final
    else:
        return best_x.reshape((T, D)), best_obj, i_final

class FKSolver:
    """
    Forward kinematics solver.
    """
    def __init__(self, gltf_file, controlled_bones=None):
        self.skeleton = load_skeleton_from_gltf(gltf_file)
        self._prepare_fk_arrays()
        self.controlled_bones = controlled_bones

        self.controlled_indices = tuple(
            i for i, name in enumerate(self.bone_names) if name in self.controlled_bones
        )
        self.default_rotations = jnp.stack(
            [jnp.eye(4, dtype=jnp.float32) for _ in self.bone_names],
            axis=0
        )
        controlled_map = -np.ones(len(self.bone_names), dtype=np.int32)
        for j, bone_idx in enumerate(self.controlled_indices):
            controlled_map[bone_idx] = 3 * j
        self.controlled_map_array = jnp.array(controlled_map, dtype=jnp.int32)

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
        return _compute_fk(
            self.local_array,
            self.parent_indices,
            self.default_rotations,
            self.controlled_indices,
            angle_vector
        )

    def get_bone_head_tail_from_fk(self, fk_transforms, bone_name):
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


class InverseKinematicsSolver:
    """
    Inverse Kinematics (IK) solver.
    """
    def __init__(self, gltf_file, controlled_bones=None, bounds=None,
                 penalty_weight=0.25, threshold=0.01, num_steps=1000, optimize_jax_cache=True):
        self.fk_solver = FKSolver(gltf_file=gltf_file, controlled_bones=controlled_bones)
        self.controlled_bones = self.fk_solver.controlled_bones

        if optimize_jax_cache:
            jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = jnp.array(lower_bounds, dtype=jnp.float32)
        self.upper_bounds = jnp.array(upper_bounds, dtype=jnp.float32)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps

        self.avg_iter_time = None

    def solve(self, target_point, initial_rotations=None, learning_rate=0.2, max_time=None,
              verbose=False, max_iterations=None, objective_functions=(), subpoints=0):
        """
        Solve the IK problem.
          - If subpoints == 0, a single configuration is optimized.
          - Otherwise, a trajectory with (subpoints+2) configurations is optimized,
            enforcing smooth transitions via additional objectives.
        Each objective function in objective_functions must have the signature:
            fn(X, fksolver, target_point)
        where X is either a single configuration (if T==1) or a trajectory (shape (T, D)).
        """
        target_point = jnp.array(target_point, dtype=jnp.float32)
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

        # Determine trajectory length: if subpoints==0, T==1 (single configuration);
        # otherwise, T = subpoints + 2 (start and final configurations).
        T = 1 if subpoints == 0 else subpoints + 2

        start_time = time.time()
        best_sol, best_obj, steps = solve_ik(
            target_point=target_point,
            init_rot=initial_rotations,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            objective_functions=objective_functions,
            fksolver=self.fk_solver,
            T=T,
            threshold=self.threshold,
            num_steps=allowed_steps,
            learning_rate=learning_rate
        )
        elapsed_time = time.time() - start_time
        if steps > 0:
            new_iter_time = elapsed_time / steps
            self.avg_iter_time = new_iter_time if self.avg_iter_time is None else 0.9 * self.avg_iter_time + 0.1 * new_iter_time

        if max_time is not None and elapsed_time > max_time and verbose:
            print(f"Time limit exceeded: {elapsed_time:.4f} seconds (allowed {max_time} seconds)")
        if best_obj > self.threshold and verbose:
            print(f"Warning: Optimization did not converge below threshold ({self.threshold}).")

        # For rendering, if a trajectory was optimized, use the final configuration.
        if T > 1:
            best_angles = best_sol[-1]
        else:
            best_angles = best_sol
        return np.array(best_angles), float(best_obj), int(steps)

    def render(self, angle_vector, target_pos=[], interactive=False):
        self.fk_solver.render(angle_vector=angle_vector, target_pos=target_pos, interactive=interactive)

    def find_optimal_learning_rate(self, target_points, lr_min=0.001, lr_max=0.5, grid_size=100, threshold=None,
                                   num_steps=None, max_time=None, stability_factor=0.75, objective_functions=()):
        if threshold is None:
            threshold = self.threshold
        if num_steps is None:
            num_steps = self.num_steps

        initial_rotations = None
        candidate_lrs = np.linspace(lr_min, lr_max, grid_size)
        results = []
        total_targets = len(target_points)
        tbar = tqdm(candidate_lrs)

        for lr in tbar:
            count_converged = 0
            time1 = time.time()
            for target in target_points:
                if initial_rotations is None:
                    init_rot = np.zeros(self.lower_bounds.shape, dtype=np.float32)
                else:
                    init_rot = initial_rotations
                _, best_obj, _ = self.solve(target_point=target, initial_rotations=init_rot,
                                            learning_rate=lr, max_time=max_time,
                                            objective_functions=objective_functions, max_iterations=num_steps)
                if best_obj < threshold:
                    count_converged += 1
            results.append((lr, count_converged, time.time()-time1))
            tbar.set_description(f"Learning rate {lr:.4f}: Converged for {count_converged}/{total_targets} targets.")
        maximum_achieved_targets = max(count for _, count, _ in results)
        full_convergence = [(lr, count, t) for lr, count, t in results if count == maximum_achieved_targets]
        full_convergence = sorted(full_convergence, key=lambda x: x[2])
        optimal_lr = full_convergence[0][0]
        print(f"Raw optimal learning rate: {optimal_lr:.4f} which converged for {maximum_achieved_targets}/{total_targets} targets.")
        print("We will adjust the optimal learning rate for stability, based on the stability factor.")
        optimal_lr *= stability_factor
        print(f"Selected optimal learning rate: {optimal_lr:.4f}")
        return optimal_lr, results


def main():
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"]
    )
    parser.add(
        "--gltf_file",
        type=str,
        default="../smplx.glb",
        help="Path to the glTF file.",
    )
    parser.add(
        "--hand",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Which hand to use (left or right).",
    )
    parser.add(
        "--bounds",
        type=str,
        default=None,
        help=(
            "List of bounds as a JSON string. "
            "For example: "
            "'[[-10, 10], [-10, 10], [-10, 10], [0, 50], [-140, 50], [-70, 25], "
            "[-45, 90], [-180, 5], [-10, 10], [-90, 90], [-70, 70], [-55, 85]]'. "
            "If not provided, defaults depend on the hand."
        ),
    )
    parser.add(
        "--controlled_bones",
        type=str,
        default=None,
        help=(
            "Comma-separated list of controlled bones. "
            "If not provided, defaults to [<hand>_collar, <hand>_shoulder, <hand>_elbow, <hand>_wrist]."
        ),
    )
    parser.add(
        "--threshold",
        type=float,
        default=0.005,
        help="Threshold value for the solver.",
    )
    parser.add(
        "--num_steps",
        type=int,
        default=500,
        help="Number of steps for the solver.",
    )
    parser.add(
        "--target_points",
        type=str,
        default=None,
        help=(
            "List of target points as a JSON string. "
            "For example: '[[0.15, 0, 0.35], [0.15, 0, 0.35], ...]'. "
            "If not provided, 110 noisy target points are generated."
        ),
    )
    parser.add(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the solver.",
    )
    parser.add(
        "--max_iterations",
        type=int,
        default=500,
        help="Maximum iterations for each solver call.",
    )
    parser.add(
        "--additional_objective_weight",
        type=float,
        default=0,
        help="Additional objective weight used in the objective functions.",
    )
    # New argument to control trajectory subpoints (0 means single configuration).
    parser.add(
        "--subpoints",
        type=int,
        default=0,
        help="Number of subpoints for trajectory IK (0 for single configuration IK).",
    )

    parser.add(
        "--cpu_only",
        action="store_true",
        help="Run on CPU only.",
    )


    args = parser.parse_args()

    if args.cpu_only:
        jax.config.update('jax_default_device', 'cpu')

    gltf_file = args.gltf_file
    hand = args.hand

    if args.bounds is None:
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
    else:
        bounds = json.loads(args.bounds)
        bounds = [tuple(b) for b in bounds]

    if args.controlled_bones is None:
        controlled_bones = [f"{hand}_collar", f"{hand}_shoulder", f"{hand}_elbow", f"{hand}_wrist"]
    else:
        controlled_bones = [b.strip() for b in args.controlled_bones.split(",")]

    if args.target_points is None:
        target = [np.array([0.15, 0, 0.35]) + np.random.normal(0, 0.05, 3) for _ in range(110)]
    else:
        target_list = json.loads(args.target_points)
        target = [np.array(pt) for pt in target_list]

    solver = InverseKinematicsSolver(
        gltf_file,
        controlled_bones=controlled_bones,
        bounds=bounds,
        threshold=args.threshold,
        num_steps=args.num_steps,
    )

    initial_rotations = None

    additional_objective_weight = args.additional_objective_weight

    # Build a list of objective functions.
    # For objectives that act on the final configuration, we use the _traj constructors.
    obj_fns = []
    # Main distance objective on the final configuration.
    obj_fns.append(distance_obj_traj(bone_name=f"{hand}_index3_look", use_head=True, weight=1.0))
    if additional_objective_weight > 0.0:
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_wrist", use_head=False, modifications=[(1, 1.0)], weight=additional_objective_weight / 3))
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_index3_look", use_head=False, modifications=[(2, 1.0)], weight=additional_objective_weight / 3))
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_shoulder", use_head=False, modifications=[(2, -1.0)], weight=(0.002 * additional_objective_weight) / 3))
    # When a trajectory is used (i.e. subpoints > 0), add smoothness and initial–pose objectives.
    if args.subpoints > 0:
        obj_fns.append(velocity_obj(weight=1e-3))
        obj_fns.append(acceleration_obj(weight=1e-3))
        obj_fns.append(jerk_obj(weight=1e-3))
        obj_fns.append(init_pose_obj(initial_rotations if initial_rotations is not None else np.zeros(solver.lower_bounds.shape, dtype=np.float32), weight=1e6))

    tbar = tqdm(range(len(target)))
    avg_steps = 0
    solved_rate = 0
    time1 = time.time()

    for i in tbar:
        time_iter = time.time()
        best_angles, obj, steps = solver.solve(
            target_point=target[i],
            initial_rotations=initial_rotations,
            learning_rate=args.learning_rate,
            max_iterations=args.max_iterations,
            objective_functions=tuple(obj_fns),
            subpoints=args.subpoints
        )
        avg_steps += steps
        tbar.set_description(f"Error: {obj:.4f} after {steps} iterations")
        if i > 10:
            print(f"Time for iteration {i}: {time.time() - time_iter:.4f} seconds. Steps: {steps}. Success: {obj < args.threshold}")
        if i == 10:
            time1 = time.time()
            avg_steps = 0

        if obj < args.threshold:
            solved_rate += 1
            print(f"Found solution for target {target[i]} after {steps} iterations")
    solved_success_rate = solved_rate / len(target)
    print("Gradient descent result: error =", obj, "after", steps, "iterations", "on average", avg_steps)
    print(f"Result: error = {obj}, iterations = {steps}, average = {avg_steps}")
    print(f"Solving {len(target)} times with total {avg_steps} iterations took {time.time() - time1:.4f} seconds")
    print(f"Success rate: {solved_success_rate:.4f}")

    # solver.render(
    #     best_angles,
    #     target_pos=[(target[i], "green")],
    #     interactive=True,
    # )

if __name__ == "__main__":
    main()
