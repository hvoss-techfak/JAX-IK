import random
import time

import numpy as np

from trac_ik import TracIK
from ur_ikfast import ur_kinematics

# IKPy is used only for the baseline IK solver and FK target generation.
import ikpy.chain

# The total number of tests.
TESTS = 10

SEED = 42

DISTANCE_ERROR = 0.01
ANGLE_ERROR = 1

# Default bounding value.
BOUND = 2 * np.pi

# Ensure consistent results.
random.seed(SEED)
np.random.seed(SEED)

robot_ikfast = ur_kinematics.URKinematics('ur5')
robot_trac_ik = TracIK(urdf_path="models/UR5.urdf", base_link_name="base_link", tip_link_name="tool0")


class IKPyRobotAdapter:
    """Tiny IKPy wrapper exposing the methods used by this benchmark.

    This keeps `additional.py` runnable without importing `llm_ik.py`.
    """

    def __init__(self, urdf_path: str, base_elements: list[str] | None = None):
        # `base_elements` can be used to skip non-chain links if needed.
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=base_elements)

    @property
    def links(self):
        return self.chain.links

    def forward_kinematics(self, lower: int, upper: int, joints: list[float]):
        """Return lists of positions and rotations for each joint in [lower, upper].

        Note: this benchmark only ever uses the final element (end-effector pose).
        Different IKPy versions expose different APIs for per-link FK, so we compute
        the full-chain FK once and replicate it for the requested joint range.
        """
        # IKPy expects a joint vector aligned with `chain.links` length.
        # Many URDFs include extra fixed/end-effector links, so we pad with zeros.
        full = [0.0] + list(joints)
        if len(full) < len(self.chain.links):
            full = full + [0.0] * (len(self.chain.links) - len(full))
        fk = self.chain.forward_kinematics(full)
        pos = fk[:3, 3]
        rot = fk[:3, :3]
        count = max(0, upper - lower + 1)
        return [pos] * count, [rot] * count

    def inverse_kinematics(self, lower: int, upper: int, target_position, target_orientation):
        """Solve IK with IKPy; return (solution, distance_error, angle_error, elapsed)."""
        target_position = np.asarray(target_position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)

        start_time = time.perf_counter()
        try:
            # IKPy uses orientation_mode="all" with a 3x3 rotation matrix.
            sol_full = self.chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                orientation_mode="all",
            )
        except TypeError:
            # Older IKPy versions use positional args.
            sol_full = self.chain.inverse_kinematics(target_position, target_orientation)
        elapsed = time.perf_counter() - start_time

        # Drop base joint; keep the first 6 actuated joints.
        solution = np.asarray(sol_full[1:7], dtype=float)
        positions, orientations = self.forward_kinematics(lower, upper, solution.tolist())
        distance = difference_distance(target_position, positions[-1])
        angle = difference_angle(target_orientation, orientations[-1])
        return solution, distance, angle, elapsed


# Create IKPy baseline/target generator.
robot_llm_ik = IKPyRobotAdapter("models/UR5.urdf")

# Cache joint limits, just in case the representations of the robot differ slightly in different implementations.
# We use IKPy's URDF bounds as the sampling range.
bounds_llm_ik: list[tuple[float, float]] = []
for link in robot_llm_ik.links:
    # Skip fixed joints and the base link.
    if getattr(link, "joint_type", None) == "fixed":
        continue
    bounds = getattr(link, "bounds", None)
    if bounds is None or bounds == (-np.inf, np.inf):
        bounds_llm_ik.append((-BOUND, BOUND))
    else:
        bounds_llm_ik.append(bounds)
# URDFs often include a base link without bounds; ensure we have 6 sampled joints.
if len(bounds_llm_ik) > 6:
    bounds_llm_ik = bounds_llm_ik[:6]

# Define data caches.
cache: dict[str, dict[str, float]] = {}
for entry in ["IKPy", "IKFast", "TRAC-IK"]:
    cache[entry] = {"Success": 0, "Elapsed": 0, "Distance": 0, "Angle": 0}

def reached(distance: float = 0, angle: float = 0) -> bool:
    """Determine if a target has been reached.

    `distance`/`angle` may be Python floats, NumPy scalars, or small NumPy arrays
    depending on which solver produced them.
    """
    distance = np.asarray(distance)
    angle = np.asarray(angle)
    # If arrays are provided, treat the worst component as the error.
    distance_v = float(distance.max()) if distance.ndim else float(distance)
    angle_v = float(angle.max()) if angle.ndim else float(angle)
    return distance_v <= DISTANCE_ERROR and angle_v <= ANGLE_ERROR


def difference_distance(a, b) -> float:
    """
    Get the difference between two positions.
    :param a: First position.
    :param b: Second position.
    :return: The differences between the two positions.
    """
    return np.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))


def difference_angle(a: float | int = 0, b: float | int = 0) -> float:
    """
    Get the difference between two angles.
    :param a: First angle.
    :param b: Second angle.
    :return: The differences between the two angle.
    """
    return sum([abs(x - y) for x, y in zip(a, b)])

def get_quaternion_angle_difference(q1, q2):
    """
    Calculates the angular difference (in degrees) between two unit quaternions.
    Assumes q1 and q2 are NumPy arrays of shape (4,) or (N, 4) for batch processing.
    :param q1: The first quaternion.
    :param q2: The second quaternion.
    :return: The angle difference.
    """
    # Calculate the dot product.
    dot_product = np.dot(q1, q2)
    # Handle batch processing (N, 4) dot (N, 4).
    if q1.ndim == 2:
        dot_product = np.sum(q1 * q2, axis=1)
    # Take the absolute value to get the shortest path.
    dot_product = np.abs(dot_product)
    # Clip the value to be in [-1.0, 1.0] to prevent acos from failing due to floating point inaccuracies.
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Calculate the angle in radians which is the half-angle.
    half_angle = np.arccos(dot_product)
    # Double it to get the full angle.
    angle_rad = 2 * half_angle
    # Convert to degrees.
    return np.degrees(angle_rad)


def quaternion_from_matrix(matrix):
    """
    Get a quaternion from a matrix.
    :param matrix: The matrix.
    :return: The quaternion.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # Symmetric matrix K.
    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0
    # Quaternion is eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def common_caching(elapsed: float, distance: float, angle: float, title: str) -> None:
    """
    Common solver caching operations.
    :param elapsed: The elapsed execution time.
    :param distance: The positional error.
    :param angle: The orientation error.
    :param title: Where to cache it.
    :return: If it was a success, the elapsed execution time, and the distance and angle offsets.
    """
    if reached(distance, angle):
        cache[title]["Success"] += 1
    else:
        cache[title]["Distance"] += distance
        cache[title]["Angle"] += angle
    cache[title]["Elapsed"] += elapsed


def ikpy_common(case: list[float]) -> tuple[list[float], list[float]]:
    """
    Apply clamping for the IKPy-based solvers.
    :param case: The joints to test for this case.
    :return: The target position and orientation to solve for.
    """
    positions, orientations = robot_llm_ik.forward_kinematics(0, 5, case)
    return positions[-1], orientations[-1]


def bench_ikpy(case: list[float]) -> None:
    """Benchmark built-in IKPy."""
    target_position, target_orientation = ikpy_common(case)
    _, distance, angle, elapsed = robot_llm_ik.inverse_kinematics(0, 5, target_position, target_orientation)
    common_caching(elapsed, distance, angle, "IKPy")


def bench_ikfast(case: list[float]) -> None:
    """Benchmark the IKFast solver."""
    t = robot_ikfast.forward(case)
    zeros = [0] * len(case)
    start_time = time.perf_counter()
    solution = robot_ikfast.inverse(t, False, zeros)
    elapsed = time.perf_counter() - start_time
    if solution is None:
        r = robot_ikfast.forward(zeros)
    else:
        r = robot_ikfast.forward(solution)
    distance = difference_distance([t[0], t[1], t[2]], [r[0], r[1], r[2]])
    angle = get_quaternion_angle_difference(
        np.array([t[3], t[4], t[5], t[6]], dtype="float64"),
        np.array([r[3], r[4], r[5], r[6]], dtype="float64"),
    )
    common_caching(elapsed, distance, angle, "IKFast")


def bench_trac_ik(case: list[float]) -> None:
    """Benchmark the Trac-IK solver."""
    lb, ub = robot_trac_ik.joint_limits
    clamped = []
    for index in range(len(case)):
        clamped.append(min(ub[index], max(case[index], lb[index])))
    t_p, t_r = robot_trac_ik.fk(np.array(clamped))
    zeros = np.zeros(len(case), dtype=float)
    start_time = time.perf_counter()
    solution = robot_trac_ik.ik(t_p, t_r, zeros)
    elapsed = time.perf_counter() - start_time
    if solution is None:
        r_p, r_r = robot_trac_ik.fk(zeros)
    else:
        r_p, r_r = robot_trac_ik.fk(solution)
    distance = difference_distance(t_p, r_p)
    angle = get_quaternion_angle_difference(quaternion_from_matrix(t_r), quaternion_from_matrix(r_r))
    common_caching(elapsed, distance, angle, "TRAC-IK")


def main() -> None:
    # Run all test cases.
    for _ in range(TESTS):
        # Generate random joints between the IKPy URDF limits.
        # Other solvers further clamp as needed if their limits differ.
        joints = [random.uniform(bounds_llm_ik[j][0], bounds_llm_ik[j][1]) for j in range(6)]

        bench_ikpy(joints)
        bench_ikfast(joints)
        bench_trac_ik(joints)

    # Tabulate data.
    s = "Solver,Success Rate (%),Average Elapsed Time (s),Average Failure Distance,Average Failure Angle (°)"
    sorted_cache = dict(sorted(cache.items(), key=lambda item: (-item[1]["Success"], item[1]["Elapsed"], item[0])))
    for entry in sorted_cache:
        failures = TESTS - sorted_cache[entry]["Success"]
        if failures < 1:
            d = "-"
            a = "-"
        else:
            d = sorted_cache[entry]["Distance"] / failures
            a = f"{sorted_cache[entry]['Angle'] / failures} °"
        s += f"\n{entry},{sorted_cache[entry]['Success'] / TESTS * 100}%,{sorted_cache[entry]['Elapsed'] / TESTS} s,{d},{a}"

        print(s)
    with open("Additional.csv", "w", encoding="utf-8", errors="ignore") as file:
        file.write(s)


if __name__ == "__main__":
    main()
