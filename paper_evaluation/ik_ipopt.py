import json
import time
import configargparse
import numpy as np

from cyipopt import Problem as IpoptProblem

from vedo import Line, Sphere, show

from helper import load_skeleton_from_gltf


def euler_to_matrix(angles):
    # Assumes angles = [angle_x, angle_y, angle_z] and returns R = R_z @ R_y @ R_x.
    angle_x, angle_y, angle_z = angles
    cx = np.cos(angle_x)
    sx = np.sin(angle_x)
    cy = np.cos(angle_y)
    sy = np.sin(angle_y)
    cz = np.cos(angle_z)
    sz = np.sin(angle_z)
    R_x = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    R_y = np.array([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    R_z = np.array([
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return R_z @ R_y @ R_x


def look_at_penalty(head, tail, target_point):
    bone_direction = tail - head
    norm_bone = np.linalg.norm(bone_direction) + 1e-6
    bone_direction = bone_direction / norm_bone
    target_direction = target_point - head
    norm_target = np.linalg.norm(target_direction) + 1e-6
    target_direction = target_direction / norm_target
    cos_theta = np.clip(np.dot(bone_direction, target_direction), -1.0, 1.0)
    misalignment_angle = np.arccos(cos_theta)
    return misalignment_angle ** 2


def compute_fk_np(local_array, parent_indices, default_rotations, controlled_indices, angle_vector):
    # Update rotations for the controlled bones using the provided angle vector.
    rotations = default_rotations.copy()
    num_controlled = len(controlled_indices)
    for j in range(num_controlled):
        bone_idx = controlled_indices[j]
        angles = angle_vector[3 * j: 3 * j + 3]
        R = euler_to_matrix(angles)
        rotations[bone_idx] = R
    n = local_array.shape[0]
    transforms = [None] * n
    for i in range(n):
        local_i = local_array[i]
        rotation_i = rotations[i]
        parent_idx = parent_indices[i]
        if parent_idx < 0:
            parent_transform = np.eye(4, dtype=np.float64)
        else:
            parent_transform = transforms[parent_idx]
        current = parent_transform @ local_i @ rotation_i
        transforms[i] = current
    return np.stack(transforms, axis=0)


def distance_obj_traj(bone_name, use_head=False, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        target_bone = head if use_head else tail
        return np.linalg.norm(target_bone - target_point) * weight

    return obj


def look_at_obj_traj(bone_name, use_head, modifications, weight):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        head, tail = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
        adjusted_target = head.copy() if use_head else tail.copy()
        for idx, delta in modifications:
            adjusted_target[idx] = adjusted_target[idx] + delta
        penalty = look_at_penalty(head, tail, adjusted_target)
        return weight * penalty

    return obj


def known_rot_obj_traj(candidate_known, mask, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        angles_reshaped = config.reshape(-1, 3)
        candidate = np.array(candidate_known, dtype=np.float64)
        mask_arr = np.array(mask, dtype=np.float64)
        diff_sq = np.sum((angles_reshaped - candidate) ** 2, axis=1)
        diff_sq = diff_sq * mask_arr
        valid_count = max(np.sum(mask_arr), 1.0)
        return np.sum(diff_sq) / valid_count * weight

    return obj


def collision_penalty_obj_traj(collider_eqs, weight=1.0):
    def obj(X, fksolver, target_point):
        config = X if X.ndim == 1 else X[-1]
        fk_transforms = fksolver.compute_fk_from_angles(config)
        joint_positions = []
        for bone_name in fksolver.bone_names:
            head, _ = fksolver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            joint_positions.append(head)
        joint_positions = np.stack(joint_positions, axis=0)
        total_penalty = 0.0
        for eq in collider_eqs:
            eq_np = np.array(eq, dtype=np.float64)
            for p in joint_positions:
                d = np.max(np.dot(eq_np[:, :3], p) + eq_np[:, 3])
                total_penalty += max(0.0, -d) ** 2
        return total_penalty * weight

    return obj


def velocity_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1:
            return 0.0
        vel = X[1:] - X[:-1]
        return weight * np.sum(vel ** 2)

    return obj


def acceleration_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1 or X.shape[0] < 3:
            return 0.0
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        return weight * np.sum(acc ** 2)

    return obj


def jerk_obj(weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1 or X.shape[0] < 4:
            return 0.0
        jerk = X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]
        return weight * np.sum(jerk ** 2)

    return obj


def init_pose_obj(init_rot, weight):
    def obj(X, fksolver, target_point):
        if X.ndim == 1:
            return 0.0
        return weight * np.sum((X[0] - init_rot) ** 2)

    return obj


class FKSolver:
    """
    Forward kinematics solver.
    """

    def __init__(self, gltf_file, controlled_bones=None):
        self.skeleton = load_skeleton_from_gltf(gltf_file)
        self._prepare_fk_arrays()
        self.controlled_bones = controlled_bones
        self.controlled_indices = [i for i, name in enumerate(self.bone_names) if name in self.controlled_bones]
        self.default_rotations = np.array([np.eye(4, dtype=np.float64) for _ in self.bone_names])

    def _prepare_fk_arrays(self):
        self.bone_names = []
        self.local_list = []
        self.parent_list = []

        def dfs(bone_name, parent_index):
            current_index = len(self.bone_names)
            self.bone_names.append(bone_name)
            bone = self.skeleton[bone_name]
            self.local_list.append(np.array(bone["local_transform"], dtype=np.float64))
            self.parent_list.append(parent_index)
            for child in bone["children"]:
                dfs(child, current_index)

        roots = [bone["name"] for bone in self.skeleton.values() if bone["parent"] is None]
        for root in roots:
            dfs(root, -1)
        self.local_array = np.stack(self.local_list, axis=0)
        self.parent_indices = np.array(self.parent_list, dtype=np.int32)

    def compute_fk_from_angles(self, angle_vector):
        return compute_fk_np(self.local_array, self.parent_indices, self.default_rotations,
                             np.array(self.controlled_indices, dtype=np.int32), angle_vector)

    def get_bone_head_tail_from_fk(self, fk_transforms, bone_name):
        try:
            idx = self.bone_names.index(bone_name)
        except ValueError:
            raise ValueError(f"Bone '{bone_name}' not found in skeleton.")
        global_transform = fk_transforms[idx]
        head = global_transform[:3, 3]
        bone = self.skeleton[bone_name]
        tail_local = np.array([0, bone["bone_length"], 0, 1], dtype=np.float64)
        tail = global_transform @ tail_local.reshape(4, 1)
        tail = tail[:3, 0]
        return head, tail

    def render(self, angle_vector=None, target_pos=[], interactive=False):
        if angle_vector is None:
            n_angles = len(self.controlled_indices) * 3
            angle_vector = np.zeros(n_angles, dtype=np.float64)
        fk_transforms = self.compute_fk_from_angles(angle_vector)
        actors = []
        for bone_name in self.bone_names:
            head, tail = self.get_bone_head_tail_from_fk(fk_transforms, bone_name)
            head_np = head
            tail_np = tail
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
                 penalty_weight=0.25, threshold=0.01, num_steps=1000, optimize_tf_cache=True):
        self.fk_solver = FKSolver(gltf_file=gltf_file, controlled_bones=controlled_bones)
        self.controlled_bones = self.fk_solver.controlled_bones

        # Set up bounds in radians.
        bounds_radians = [(np.radians(l), np.radians(h)) for l, h in bounds]
        lower_bounds, upper_bounds = zip(*bounds_radians)
        self.lower_bounds = np.array(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.array(upper_bounds, dtype=np.float64)

        self.penalty_weight = penalty_weight
        self.threshold = threshold
        self.num_steps = num_steps

        self.avg_iter_time = None

    def solve(self, target_point, initial_rotations=None, learning_rate=0.2, max_time=None,
              verbose=False, max_iterations=None, objective_functions=(), subpoints=0):
        """
        Solve the IK problem using IPOPT.
          - If subpoints == 0, a single configuration is optimized.
          - Otherwise, a trajectory with (subpoints+2) configurations is optimized.
        Each objective function must have signature: fn(X, fksolver, target_point)
        where X is a single configuration (if T==1) or a trajectory (shape (T, D)).
        """
        target_point = np.array(target_point, dtype=np.float64)
        if initial_rotations is None:
            initial_rotations = np.zeros(self.lower_bounds.shape, dtype=np.float64)
        initial_rotations = np.array(initial_rotations, dtype=np.float64)

        allowed_steps = self.num_steps if max_iterations is None else max_iterations

        # Determine trajectory length.
        T = 1 if subpoints == 0 else subpoints + 2

        if T == 1:
            x0 = initial_rotations.copy()
            lower_bounds_traj = self.lower_bounds.copy()
            upper_bounds_traj = self.upper_bounds.copy()
        else:
            x0 = np.tile(initial_rotations, T)
            lower_bounds_traj = np.tile(self.lower_bounds, T)
            upper_bounds_traj = np.tile(self.upper_bounds, T)

        def total_objective(x_flat):
            if T == 1:
                X = x_flat
            else:
                X = x_flat.reshape(T, -1)
            total = 0.0
            for fn in objective_functions:
                total += fn(X, self.fk_solver, target_point)
            return total

        # Define a problem object for IPOPT.
        class IKProblem:
            def __init__(self, n, obj_func,threshold):
                self.n = n
                self.obj_func = obj_func
                self.threshold = threshold
                self.current_x = None
                self.steps = 0

            def objective(self, x):
                self.current_x = x
                return self.obj_func(x)

            def gradient(self, x):
                # Finite difference approximation.
                grad = np.zeros_like(x)
                eps = 1e-8
                f0 = self.obj_func(x)
                for i in range(len(x)):
                    x_eps = x.copy()
                    x_eps[i] += eps
                    grad[i] = (self.obj_func(x_eps) - f0) / eps
                return grad

            def constraints(self, x):
                return np.array([])  # No constraints

            def jacobian(self, x):
                return np.array([])

            def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                             regularization_size, alpha_du, alpha_pr, ls_trials):
                # Stop early if the objective falls below the threshold.
                self.steps += 1
                if self.current_x is not None:
                    obj_value = self.obj_func(self.current_x)
                    if obj_value < self.threshold:
                        return False
                return True

        n = len(x0)
        problem = IKProblem(n, total_objective,self.threshold)
        nlp = IpoptProblem(n=n, m=0, problem_obj=problem,
                           lb=lower_bounds_traj, ub=upper_bounds_traj, cl=[], cu=[])
        # Use IPOPT’s finite-difference Jacobian approximation.
        nlp.addOption("max_iter", allowed_steps)
        nlp.addOption("print_level", 0)
        start_time = time.time()
        x_opt, info = nlp.solve(x0)
        elapsed_time = time.time() - start_time
        steps_val = problem.steps

        best_obj = total_objective(x_opt)
        if max_time is not None and elapsed_time > max_time and verbose:
            print(f"Time limit exceeded: {elapsed_time:.4f} sec (allowed {max_time} sec)")
        if best_obj > self.threshold and verbose:
            print(f"Warning: Optimization did not converge below threshold ({self.threshold}).")

        if T > 1:
            best_angles = x_opt.reshape(T, -1)[-1]
        else:
            best_angles = x_opt
        return best_angles, best_obj, steps_val

    def render(self, angle_vector, target_pos=[], interactive=False):
        self.fk_solver.render(angle_vector=angle_vector, target_pos=target_pos, interactive=interactive)


def main():
    parser = configargparse.ArgumentParser(
        description="Inverse Kinematics Solver Configuration",
        default_config_files=["config.ini"]
    )
    parser.add("--gltf_file", type=str, default="../smplx.glb",
               help="Path to the glTF file.")
    parser.add("--hand", type=str, choices=["left", "right"], default="left",
               help="Which hand to use (left or right).")
    parser.add("--bounds", type=str, default=None,
               help=("List of bounds as a JSON string. "
                     "For example: "
                     "'[[-10, 10], [-10, 10], [-10, 10], [0, 50], [-140, 50], [-70, 25], "
                     "[-45, 90], [-180, 5], [-10, 10], [-90, 90], [-70, 70], [-55, 85]]'. "
                     "If not provided, defaults depend on the hand."))
    parser.add("--controlled_bones", type=str, default=None,
               help=("Comma-separated list of controlled bones. "
                     "If not provided, defaults to [<hand>_collar, <hand>_shoulder, <hand>_elbow, <hand>_wrist]."))
    parser.add("--threshold", type=float, default=0.005,
               help="Threshold value for the solver.")
    parser.add("--num_steps", type=int, default=500,
               help="Number of steps for the solver.")
    parser.add("--target_points", type=str, default=None,
               help=("List of target points as a JSON string. "
                     "For example: '[[0.15, 0, 0.35], [0.15, 0, 0.35], ...]'. "
                     "If not provided, 110 noisy target points are generated."))
    parser.add("--learning_rate", type=float, default=0.1,
               help="Learning rate for the solver.")
    parser.add("--max_iterations", type=int, default=500,
               help="Maximum iterations for each solver call.")
    parser.add("--additional_objective_weight", type=float, default=0,
               help="Additional objective weight used in the objective functions.")
    parser.add("--subpoints", type=int, default=0,
               help="Number of subpoints for trajectory IK (0 for single configuration IK).")
    parser.add("--cpu_only", action="store_true",
               help="Run on CPU only.")

    args = parser.parse_args()

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

    # Build objective functions.
    obj_fns = []
    obj_fns.append(distance_obj_traj(bone_name=f"{hand}_index3_look", use_head=True, weight=1.0))
    if additional_objective_weight > 0.0:
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_wrist", use_head=False, modifications=[(1, 1.0)], weight=additional_objective_weight / 3))
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_index3_look", use_head=False, modifications=[(2, 1.0)], weight=additional_objective_weight / 3))
        obj_fns.append(look_at_obj_traj(bone_name=f"{hand}_shoulder", use_head=False, modifications=[(2, -1.0)], weight=(0.002 * additional_objective_weight) / 3))
    if args.subpoints > 0:
        obj_fns.append(velocity_obj(weight=1e-3))
        obj_fns.append(acceleration_obj(weight=1e-3))
        obj_fns.append(jerk_obj(weight=1e-3))
        init_rot = initial_rotations if initial_rotations is not None else np.zeros(solver.lower_bounds.shape, dtype=np.float64)
        obj_fns.append(init_pose_obj(np.array(init_rot, dtype=np.float64), weight=1e6))

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
            subpoints=args.subpoints
        )
        avg_steps += steps
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

    # Uncomment below to render the final result:
    # solver.render(
    #     best_angles,
    #     target_pos=[(target[i], "green")],
    #     interactive=True,
    # )


if __name__ == "__main__":
    main()
