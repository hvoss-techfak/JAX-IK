import json
import time

import configargparse
import numpy as np
import tensorflow as tf
from helper import load_skeleton_from_gltf
from vedo import Line, Sphere, show


def tf_euler_to_matrix(angles):
    # Assumes angles = [angle_x, angle_y, angle_z] and returns R = R_z @ R_y @ R_x.
    cx = tf.cos(angles[0])
    sx = tf.sin(angles[0])
    cy = tf.cos(angles[1])
    sy = tf.sin(angles[1])
    cz = tf.cos(angles[2])
    sz = tf.sin(angles[2])
    R_x = tf.stack(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cx, -sx, 0.0],
            [0.0, sx, cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        axis=0,
    )
    R_y = tf.stack(
        [
            [cy, 0.0, sy, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sy, 0.0, cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        axis=0,
    )
    R_z = tf.stack(
        [
            [cz, -sz, 0.0, 0.0],
            [sz, cz, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        axis=0,
    )
    return tf.matmul(tf.matmul(R_z, R_y), R_x)


def tf_look_at_penalty(head, tail, target_point):
    bone_direction = tail - head
    bone_direction = bone_direction / (tf.norm(bone_direction) + 1e-6)
    target_direction = target_point - head
    target_direction = target_direction / (tf.norm(target_direction) + 1e-6)
    # Use tensordot with axes=1 to get the dot product.
    cos_theta = tf.clip_by_value(
        tf.tensordot(bone_direction, target_direction, axes=1), -1.0, 1.0
    )
    misalignment_angle = tf.acos(cos_theta)
    return misalignment_angle**2


def tf_matrix_to_euler(R):
    r11 = R[0, 0]
    r12 = R[0, 1]
    r13 = R[0, 2]
    r21 = R[1, 0]
    r22 = R[1, 1]
    r23 = R[1, 2]
    r31 = R[2, 0]
    angle_y = -tf.asin(tf.clip_by_value(r31, -1.0, 1.0))
    angle_x = tf.atan2(R[2, 1], R[2, 2])
    angle_z = tf.atan2(R[1, 0], R[0, 0])
    return tf.stack([angle_x, angle_y, angle_z])


def tf_rotation_matrix_from_axis_angle(axis, angle):
    c = tf.cos(angle)
    s = tf.sin(angle)
    t = 1 - c
    x, y, z = axis[0], axis[1], axis[2]
    R3 = tf.stack(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        axis=0,
    )
    R4 = tf.eye(4, dtype=tf.float32)
    R4 = tf.tensor_scatter_nd_update(
        R4,
        indices=[
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
        updates=tf.reshape(R3, [-1]),
    )
    return R4


def _compute_fk_tf(
    local_array, parent_indices, default_rotations, controlled_indices, angle_vector
):
    rotations = default_rotations
    num_controlled = int(controlled_indices.shape[0])
    for j in tf.range(num_controlled):
        bone_idx = controlled_indices[j]
        angles = angle_vector[3 * j : 3 * j + 3]
        R = tf_euler_to_matrix(angles)
        rotations = tf.tensor_scatter_nd_update(
            rotations, indices=[[bone_idx]], updates=[R]
        )
    # Use a static number of iterations (assumed known from the tf.stack)
    n = int(local_array.shape[0])
    carry = tf.zeros([n, 4, 4], dtype=tf.float32)
    i = tf.constant(0)

    def cond(i, carry):
        return i < n

    def body(i, carry):
        local_i = local_array[i]
        rotation_i = rotations[i]
        parent_idx = parent_indices[i]
        parent_transform = tf.cond(
            parent_idx < 0,
            lambda: tf.eye(4, dtype=tf.float32),
            lambda: carry[parent_idx],
        )
        current = tf.matmul(tf.matmul(parent_transform, local_i), rotation_i)
        carry = tf.tensor_scatter_nd_update(carry, indices=[[i]], updates=[current])
        return [i + 1, carry]  # Return a list to match the initial structure.

    _, final_carry = tf.while_loop(cond, body, [i, carry], maximum_iterations=n)
    return final_carry


def distance_obj_traj(bone_name, use_head=False, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if len(tf.shape(X)) == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        target_bone = head if use_head else tail
        return tf.norm(target_bone - target_point) * weight

    return obj


def look_at_obj_traj(bone_name, use_head, modifications, weight):
    def obj(X, fksolver, target_point):
        config = X if len(tf.shape(X)) == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        adjusted_target = head if use_head else tail
        for idx, delta in modifications:
            adjusted_target = tf.tensor_scatter_nd_update(
                adjusted_target, indices=[[idx]], updates=[adjusted_target[idx] + delta]
            )
        penalty = tf_look_at_penalty(head, tail, adjusted_target)
        return weight * penalty

    return obj


def known_rot_obj_traj(candidate_known, mask, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if len(tf.shape(X)) == 1 else X[-1]
        angles_reshaped = tf.reshape(config, [-1, 3])
        candidate = tf.convert_to_tensor(candidate_known, dtype=tf.float32)
        mask_arr = tf.convert_to_tensor(mask, dtype=tf.float32)
        diff_sq = tf.reduce_sum(tf.square(angles_reshaped - candidate), axis=1)
        diff_sq = diff_sq * mask_arr
        valid_count = tf.maximum(tf.reduce_sum(mask_arr), 1.0)
        return tf.reduce_sum(diff_sq) / valid_count * weight

    return obj


def collision_penalty_obj_traj(collider_eqs, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if len(tf.shape(X)) == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        joint_positions = []
        for bone_name in fksolver.bone_names:
            head, _ = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            joint_positions.append(head)
        joint_positions = tf.stack(joint_positions, axis=0)
        total_penalty = 0.0
        for eq in collider_eqs:
            eq_tf = tf.convert_to_tensor(eq, dtype=tf.float32)

            def point_penalty(p):
                d = tf.reduce_max(tf.tensordot(eq_tf[:, :3], p, axes=1) + eq_tf[:, 3])
                return tf.square(tf.maximum(0.0, -d))

            penalties = tf.map_fn(point_penalty, joint_positions)
            total_penalty += tf.reduce_sum(penalties)
        return total_penalty * weight

    return obj


# Smoothness and initial pose objectives.
def velocity_obj(weight):
    def obj(X, fksolver, target_point):
        if len(tf.shape(X)) == 1:
            return 0.0
        vel = X[1:] - X[:-1]
        return weight * tf.reduce_sum(tf.square(vel))

    return obj


def acceleration_obj(weight):
    def obj(X, fksolver, target_point):
        if len(tf.shape(X)) == 1 or tf.shape(X)[0] < 3:
            return 0.0
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        return weight * tf.reduce_sum(tf.square(acc))

    return obj


def jerk_obj(weight):
    def obj(X, fksolver, target_point):
        if len(tf.shape(X)) == 1 or tf.shape(X)[0] < 4:
            return 0.0
        jerk = X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]
        return weight * tf.reduce_sum(tf.square(jerk))

    return obj


def init_pose_obj(init_rot, weight):
    def obj(X, fksolver, target_point):
        if len(tf.shape(X)) == 1:
            return 0.0
        return weight * tf.reduce_sum(tf.square(X[0] - init_rot))

    return obj


def solve_ik(
    target_point,
    init_rot,
    lower_bounds,
    upper_bounds,
    objective_functions,
    fksolver,
    T,
    threshold=0.01,
    num_steps=1000,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    D = tf.shape(init_rot)[0]
    # Prepare initial variable and bounds.
    if T == 1:
        x0 = init_rot  # shape (D,)
        lower_bounds_traj = lower_bounds
        upper_bounds_traj = upper_bounds
    else:
        x0 = tf.tile(init_rot, [T])  # flattened trajectory of shape (T*D,)
        lower_bounds_traj = tf.tile(lower_bounds, [T])
        upper_bounds_traj = tf.tile(upper_bounds, [T])

    def total_objective(x_flat):
        if T == 1:
            X = x_flat
        else:
            X = tf.reshape(x_flat, [T, D])
        total = 0.0
        for fn in objective_functions:
            total += fn(X, fksolver, target_point)
        return total

    # Initialize Adam states.
    m = tf.zeros_like(x0)
    v = tf.zeros_like(x0)
    best_x = x0
    best_obj = total_objective(x0)

    # Optimization loop: we unroll a fixed number of iterations.
    for i in tf.range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(x0)
            loss = total_objective(x0)
        grad_val = tape.gradient(loss, x0)
        m = beta1 * m + (1 - beta1) * grad_val
        v = beta2 * v + (1 - beta2) * tf.square(grad_val)
        m_hat = m / (1 - tf.pow(beta1, tf.cast(i + 1, tf.float32)))
        v_hat = v / (1 - tf.pow(beta2, tf.cast(i + 1, tf.float32)))
        x_new = x0 - learning_rate * m_hat / (tf.sqrt(v_hat) + epsilon)
        x_new = tf.clip_by_value(x_new, lower_bounds_traj, upper_bounds_traj)
        loss_new = total_objective(x_new)
        best_x = tf.cond(loss_new < best_obj, lambda: x_new, lambda: best_x)
        best_obj = tf.minimum(loss_new, best_obj)
        if best_obj < threshold:
            num_steps = i + 1
            break
        x0 = x_new
    if T == 1:
        return best_x, best_obj, num_steps
    else:
        return tf.reshape(best_x, [T, D]), best_obj, num_steps


class FKSolver:
    """
    Forward kinematics solver.
    """

    def __init__(self, gltf_file, controlled_bones=None):
        self.skeleton = load_skeleton_from_gltf(gltf_file)
        self._prepare_fk_arrays()
        self.controlled_bones = controlled_bones
        self.controlled_indices = [
            i for i, name in enumerate(self.bone_names) if name in self.controlled_bones
        ]
        self.default_rotations = tf.stack(
            [tf.eye(4, dtype=tf.float32) for _ in self.bone_names], axis=0
        )
        controlled_map = -np.ones(len(self.bone_names), dtype=np.int32)
        for j, bone_idx in enumerate(self.controlled_indices):
            controlled_map[bone_idx] = 3 * j
        self.controlled_map_array = tf.convert_to_tensor(controlled_map, dtype=tf.int32)

    def _prepare_fk_arrays(self):
        self.bone_names = []
        self.local_list = []
        self.parent_list = []

        def dfs(bone_name, parent_index):
            current_index = len(self.bone_names)
            self.bone_names.append(bone_name)
            bone = self.skeleton[bone_name]
            self.local_list.append(
                tf.convert_to_tensor(bone["local_transform"], dtype=tf.float32)
            )
            self.parent_list.append(parent_index)
            for child in bone["children"]:
                dfs(child, current_index)

        roots = [
            bone["name"] for bone in self.skeleton.values() if bone["parent"] is None
        ]
        for root in roots:
            dfs(root, -1)
        self.local_array = tf.stack(self.local_list, axis=0)
        self.parent_indices = tf.convert_to_tensor(self.parent_list, dtype=tf.int32)

    def compute_fk_from_angles(self, angle_vector):
        return _compute_fk_tf(
            self.local_array,
            self.parent_indices,
            self.default_rotations,
            tf.convert_to_tensor(self.controlled_indices, dtype=tf.int32),
            angle_vector,
        )

    def get_bone_head_tail_from_fk(self, fk_transforms, bone_name):
        try:
            idx = self.bone_names.index(bone_name)
        except ValueError:
            raise ValueError(f"Bone '{bone_name}' not found in skeleton.")
        global_transform = fk_transforms[idx]
        head = global_transform[:3, 3]
        bone = self.skeleton[bone_name]
        tail_local = tf.convert_to_tensor(
            [0, bone["bone_length"], 0, 1], dtype=tf.float32
        )
        tail = tf.matmul(global_transform, tf.reshape(tail_local, [4, 1]))
        tail = tf.reshape(tail[:3], [-1])
        return head, tail

    def render(self, angle_vector=None, target_pos=[], interactive=False):
        if angle_vector is None:
            n_angles = len(self.controlled_indices) * 3
            angle_vector = tf.zeros([n_angles], dtype=tf.float32)
        if not isinstance(angle_vector, tf.Tensor):
            angle_vector = tf.convert_to_tensor(angle_vector, dtype=tf.float32)
        fk_transforms = self.compute_fk_from_angles(angle_vector)
        actors = []
        for bone_name in self.bone_names:
            head, tail = self.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            head_np = head.numpy()
            tail_np = tail.numpy()
            actors.append(Line(head_np, tail_np, lw=3, c="blue"))
            actors.append(Sphere(pos=head_np, r=0.02, c="red"))
        for target, color in target_pos:
            actors.append(
                Sphere(pos=target, r=0.02 if color == "green" else 0.01, c=color)
            )
        show(actors, "Skeleton FK", axes=1, interactive=interactive)


class InverseKinematicsSolver:
    """
    Inverse Kinematics (IK) solver.
    """

    def __init__(
        self,
        gltf_file,
        controlled_bones=None,
        bounds=None,
        penalty_weight=0.25,
        threshold=0.01,
        num_steps=1000,
        optimize_tf_cache=True,
    ):
        self.fk_solver = FKSolver(
            gltf_file=gltf_file, controlled_bones=controlled_bones
        )
        self.controlled_bones = self.fk_solver.controlled_bones

        # Set up bounds in radians.
        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
        self.upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps

        self.avg_iter_time = None

    def solve(
        self,
        target_point,
        initial_rotations=None,
        learning_rate=0.2,
        max_time=None,
        verbose=False,
        max_iterations=None,
        objective_functions=(),
        subpoints=0,
    ):
        """
        Solve the IK problem.
          - If subpoints == 0, a single configuration is optimized.
          - Otherwise, a trajectory with (subpoints+2) configurations is optimized.
        Each objective function must have signature: fn(X, fksolver, target_point)
        where X is a single configuration (if T==1) or a trajectory (shape (T, D)).
        """
        target_point = tf.convert_to_tensor(target_point, dtype=tf.float32)
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float32)
        initial_rotations = tf.convert_to_tensor(initial_rotations, dtype=tf.float32)

        allowed_steps = self.num_steps if max_iterations is None else max_iterations

        # Determine trajectory length.
        T = 1 if subpoints == 0 else subpoints + 2

        start_time = time.time()
        best_sol, best_obj, steps = solve_ik(
            target_point,
            initial_rotations,
            self.lower_bounds,
            self.upper_bounds,
            objective_functions,
            self.fk_solver,
            T,
            threshold=self.threshold,
            num_steps=allowed_steps,
            learning_rate=learning_rate,
        )
        elapsed_time = time.time() - start_time
        steps_val = int(steps.numpy())  # convert the tensor to a Python int
        if steps_val > 0:
            new_iter_time = elapsed_time / steps_val
            if self.avg_iter_time is None:
                self.avg_iter_time = new_iter_time
            else:
                self.avg_iter_time = 0.9 * self.avg_iter_time + 0.1 * new_iter_time

        if max_time is not None and elapsed_time > max_time and verbose:
            print(
                f"Time limit exceeded: {elapsed_time:.4f} sec (allowed {max_time} sec)"
            )
        if best_obj > self.threshold and verbose:
            print(
                f"Warning: Optimization did not converge below threshold ({self.threshold})."
            )

        if T > 1:
            best_angles = best_sol[-1]
        else:
            best_angles = best_sol
        return best_angles.numpy(), float(best_obj), int(steps)

    def render(self, angle_vector, target_pos=[], interactive=False):
        self.fk_solver.render(
            angle_vector=angle_vector, target_pos=target_pos, interactive=interactive
        )


def main():
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"],
    )
    parser.add(
        "--gltf_file", type=str, default="../smplx.glb", help="Path to the glTF file."
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
        "--threshold", type=float, default=0.005, help="Threshold value for the solver."
    )
    parser.add(
        "--num_steps", type=int, default=500, help="Number of steps for the solver."
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
        "--learning_rate", type=float, default=0.1, help="Learning rate for the solver."
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
        tf.config.set_visible_devices([], "GPU")

    gltf_file = args.gltf_file
    hand = args.hand

    if args.bounds is None:
        if hand == "left":
            bounds = [
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (0, 50),
                (-140, 50),
                (-70, 25),
                (-45, 90),
                (-180, 5),
                (-10, 10),
                (-90, 90),
                (-70, 70),
                (-55, 85),
            ]
        else:
            bounds = [
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (-50, 0),
                (-50, 140),
                (-25, 70),
                (-90, 45),
                (-5, 180),
                (-10, 10),
                (-90, 90),
                (-70, 70),
                (-55, 85),
            ]
    else:
        bounds = json.loads(args.bounds)
        bounds = [tuple(b) for b in bounds]

    if args.controlled_bones is None:
        controlled_bones = [
            f"{hand}_collar",
            f"{hand}_shoulder",
            f"{hand}_elbow",
            f"{hand}_wrist",
        ]
    else:
        controlled_bones = [b.strip() for b in args.controlled_bones.split(",")]

    if args.target_points is None:
        target = [
            np.array([0.15, 0, 0.35]) + np.random.normal(0, 0.05, 3) for _ in range(110)
        ]
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

    # Build objective functions.
    obj_fns = []
    obj_fns.append(
        distance_obj_traj(bone_name=f"{hand}_index3_look", use_head=True, weight=1.0)
    )
    if additional_objective_weight > 0.0:
        obj_fns.append(
            look_at_obj_traj(
                bone_name=f"{hand}_wrist",
                use_head=False,
                modifications=[(1, 1.0)],
                weight=additional_objective_weight / 3,
            )
        )
        obj_fns.append(
            look_at_obj_traj(
                bone_name=f"{hand}_index3_look",
                use_head=False,
                modifications=[(2, 1.0)],
                weight=additional_objective_weight / 3,
            )
        )
        obj_fns.append(
            look_at_obj_traj(
                bone_name=f"{hand}_shoulder",
                use_head=False,
                modifications=[(2, -1.0)],
                weight=(0.002 * additional_objective_weight) / 3,
            )
        )
    if args.subpoints > 0:
        obj_fns.append(velocity_obj(weight=1e-3))
        obj_fns.append(acceleration_obj(weight=1e-3))
        obj_fns.append(jerk_obj(weight=1e-3))
        # Use initial_rotations if provided; otherwise zeros.
        init_rot = (
            initial_rotations
            if initial_rotations is not None
            else np.zeros(solver.lower_bounds.shape, dtype=np.float32)
        )
        obj_fns.append(
            init_pose_obj(tf.convert_to_tensor(init_rot, dtype=tf.float32), weight=1e6)
        )

    avg_steps = 0
    solved_rate = 0

    time1 = time.time()

    for i in range(len(target)):
        time_iter = time.time()
        best_angles, obj, steps = solver.solve(
            target_point=target[i],
            initial_rotations=initial_rotations,
            learning_rate=args.learning_rate,
            max_iterations=args.max_iterations,
            objective_functions=tuple(obj_fns),
            subpoints=args.subpoints,
        )
        avg_steps += steps
        if i > 10:
            print(
                f"Time for iteration {i}: {time.time() - time_iter:.4f} seconds. Steps: {steps}. Success: {obj < args.threshold}"
            )

        if i == 10:
            time1 = time.time()
            avg_steps = 0

        if obj < args.threshold:
            solved_rate += 1
            print(f"Found solution for target {target[i]} after {steps} iterations")
    solved_success_rate = solved_rate / len(target)
    print(
        "Gradient descent result: error =",
        obj,
        "after",
        steps,
        "iterations",
        "on average",
        avg_steps,
    )
    print(f"Result: error = {obj}, iterations = {steps}, average = {avg_steps}")
    print(
        f"Solving {len(target)} times with total {avg_steps} iterations took {time.time() - time1:.4f} seconds"
    )
    print(f"Success rate: {solved_success_rate:.4f}")

    # solver.render(
    #     best_angles,
    #     target_pos=[(target[i], "green")],
    #     interactive=True,
    # )


if __name__ == "__main__":
    main()
