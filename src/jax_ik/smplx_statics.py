import numpy as np
import math

# fmt: off
gltf_file = "smplx.glb"

body_bones = {
    0: "left_hip",
    1: "right_hip",
    2: "spine1",
    3: "left_knee",
    4: "right_knee",
    5: "spine2",
    6: "left_ankle",
    7: "right_ankle",
    8: "spine3",
    9: "left_foot",
    10: "right_foot",
    11: "neck",
    12: "left_collar",
    13: "right_collar",
    14: "head",
    15: "left_shoulder",
    16: "right_shoulder",
    17: "left_elbow",
    18: "right_elbow",
    19: "left_wrist",
    20: "right_wrist",
}

left_hand_bones = {
    0: "left_index1",
    1: "left_index2",
    2: "left_index3",
    3: "left_middle1",
    4: "left_middle2",
    5: "left_middle3",
    6: "left_pinky1",
    7: "left_pinky2",
    8: "left_pinky3",
    9: "left_ring1",
    10: "left_ring2",
    11: "left_ring3",
    12: "left_thumb1",
    13: "left_thumb2",
    14: "left_thumb3",
}

right_hand_bones = {
    0: "right_index1",
    1: "right_index2",
    2: "right_index3",
    3: "right_middle1",
    4: "right_middle2",
    5: "right_middle3",
    6: "right_pinky1",
    7: "right_pinky2",
    8: "right_pinky3",
    9: "right_ring1",
    10: "right_ring2",
    11: "right_ring3",
    12: "right_thumb1",
    13: "right_thumb2",
    14: "right_thumb3",
}

left_arm_bounds_dict = {
    "left_collar": ([-10, -10, -10], [10, 10, 10]),
    "left_shoulder": ([-120, -140, -65], [70, 50, 25]),
    "left_elbow": ([-100, -180, -10], [90, 10, 10]),
    "left_wrist": ([-120, -70, -70], [90, 60, 80]),
    
    "left_thumb1": ([-20, -30, -30], [90, 30, 30]),
    "left_thumb2": ([-5, -40, -5], [5, 10, 5]),
    "left_thumb3": ([-5, -5, -5], [5, 60, 5]),
    "left_index1": ([-5, -10, -90], [5, 10, 10]),
    "left_index2": ([-5, -10, -90], [5, 10, 10]),
    "left_index3": ([-5, -10, -90], [5, 10, 10]),
    "left_middle1": ([-5, -10, -90], [5, 10, 10]),
    "left_middle2": ([-5, -10, -90], [5, 10, 10]),
    "left_middle3": ([-5, -10, -90], [5, 10, 10]),
    "left_ring1": ([-5, -10, -90], [5, 10, 10]),
    "left_ring2": ([-5, -10, -90], [5, 10, 10]),
    "left_ring3": ([-5, -10, -90], [5, 10, 10]),
    "left_pinky1": ([-5, -10, -90], [5, 10, 10]),
    "left_pinky2": ([-5, -10, -90], [5, 10, 10]),
    "left_pinky3": ([-5, -10, -90], [5, 10, 10]),
}

left_full_body_bounds_dict = {
    "left_eye_smplhf": ([-15, -15, -15],[15, 15, 15]),
    "head": ([-15, -15, -15], [15, 15, 15]),
    "jaw": ([-15, -15, -15], [15, 15, 15]),
    "neck": ([-15, -15, -15], [15, 15, 15]),
    "spine3": ([-15, -15, -15], [15, 15, 15]),
    "spine2": ([-15, -15, -15], [15, 15, 15]),
    "spine1": ([-15, -15, -15], [15, 15, 15]),
    "pelvis": ([-15, -15, -15], [15, 15, 15]),
    "left_hip": ([-15, -15, -15], [15, 15, 15]),
    "left_knee": ([-15, -15, -15], [15, 15, 15]),
    "left_ankle": ([-15, -15, -15], [15, 15, 15]),
    "left_foot": ([-15, -15, -15], [15, 15, 15]),
}
left_full_body_bounds_dict = {**left_full_body_bounds_dict, **left_arm_bounds_dict}



right_bound_inverse = [1,-1,-1]


def mirror_limits(left_bound_dicts: dict) -> dict:
    """
    Mirror joint limits from the left side to the right side of the body.

    Args:
        left_bound_dicts (dict): Dictionary of left-side joint limits, mapping bone names to (min, max) tuples.

    Returns:
        dict: Mirrored joint limits for the right side, with axes inverted as appropriate.
    """
    out_bound_dicts = {}
    for k, v in left_bound_dicts.items():
        if "left_" in k:
            k = k.replace("left_", "right_")
            x_min = v[0][0] * right_bound_inverse[0]
            x_max = v[1][0] * right_bound_inverse[0]
            y_min = v[0][1] * right_bound_inverse[1]
            y_max = v[1][1] * right_bound_inverse[1]
            z_min = v[0][2] * right_bound_inverse[2]
            z_max = v[1][2] * right_bound_inverse[2]

            # swap the min and max if needed
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            if z_min > z_max:
                z_min, z_max = z_max, z_min
            out_bound_dicts[k] = ([x_min, y_min, z_min], [x_max, y_max, z_max])
    return out_bound_dicts


right_arm_bounds_dict = mirror_limits(left_arm_bounds_dict)
right_full_body_bounds_dict = mirror_limits(left_full_body_bounds_dict)
complete_full_body_bounds_dict = {
    **left_full_body_bounds_dict,
    **right_full_body_bounds_dict,
}
    

def bounds_dict_to_list(bounds_dict: dict, hand_prefix: str) -> tuple[list, list]:
    """
    Convert a dictionary of joint bounds to a flat list and a list of controlled bone names.

    Args:
        bounds_dict (dict): Dictionary mapping bone names to (min, max) tuples.
        hand_prefix (str): Prefix for the hand ("left" or "right").

    Returns:
        tuple: (flat_bounds, controlled_bones)
            - flat_bounds (list): List of (min, max) tuples for each joint angle in order.
            - controlled_bones (list): List of bone names in the same order.
    """
    # Define the bone order as in the original lists
    bone_order = [
        "collar",
        "shoulder",
        "elbow",
        "wrist",
        "thumb1",
        "thumb2",
        "thumb3",
        "index1",
        "index2",
        "index3",
        "middle1",
        "middle2",
        "middle3",
        "ring1",
        "ring2",
        "ring3",
        "pinky1",
        "pinky2",
        "pinky3",
    ]

    flat_bounds = []
    controlled_bones = []
    for bone in bone_order:
        bone_name = f"{hand_prefix}_{bone}"
        controlled_bones.append(bone_name)
        lower, upper = bounds_dict[bone_name]
        for i in range(3):
            flat_bounds.append((lower[i], upper[i]))
    return flat_bounds, controlled_bones


left_bounds, controlled_bones_left = bounds_dict_to_list(left_arm_bounds_dict, "left")
right_bounds, controlled_bones_right = bounds_dict_to_list(right_arm_bounds_dict, "right")


# combine both lists
controlled_bones_both = controlled_bones_left.copy()
controlled_bones_both += [b for b in controlled_bones_right if b not in controlled_bones_both]



def get_pointing_pose() -> dict:
    """
    Get the default joint angles for a pointing hand pose for both left and right hands.

    Returns:
        dict: Mapping from bone names to numpy arrays of joint angles (in radians).
    """
    default_pose = {}
    default_pose["left_thumb1"] = np.array([math.radians(-45), math.radians(45), math.radians(-45)], dtype=np.float32)
    default_pose["left_thumb2"] = np.array([math.radians(0), math.radians(45), math.radians(0)], dtype=np.float32)
    default_pose["left_thumb3"] = np.array([math.radians(0), math.radians(10), math.radians(0)], dtype=np.float32)
    default_pose["left_index1"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_index2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_index3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_middle1"] = np.array([math.radians(-10), math.radians(0), math.radians(-80)], dtype=np.float32)
    default_pose["left_middle2"] = np.array([math.radians(0), math.radians(0), math.radians(-80)], dtype=np.float32)
    default_pose["left_middle3"] = np.array([math.radians(0), math.radians(0), math.radians(-50)], dtype=np.float32)
    default_pose["left_ring1"] = np.array([math.radians(-10), math.radians(0), math.radians(-80)], dtype=np.float32)
    default_pose["left_ring2"] = np.array([math.radians(0), math.radians(0), math.radians(-80)], dtype=np.float32)
    default_pose["left_ring3"] = np.array([math.radians(0), math.radians(0), math.radians(-50)], dtype=np.float32)
    default_pose["left_pinky1"] = np.array([math.radians(-30), math.radians(-10), math.radians(-80)], dtype=np.float32)
    default_pose["left_pinky2"] = np.array([math.radians(0), math.radians(-50), math.radians(-80)], dtype=np.float32)
    default_pose["left_pinky3"] = np.array([math.radians(0), math.radians(0), math.radians(-50)], dtype=np.float32)
    
    right_inverse = np.asarray([1, -1, -1])
    for k in list(default_pose.keys()):
        k1 = k.replace("left_", "right_")
        default_pose[k1] = default_pose[k] * right_inverse

    return default_pose


def get_shaping_pose() -> dict:
    """
    Get the default joint angles for a shaping hand pose for both left and right hands.

    Returns:
        dict: Mapping from bone names to numpy arrays of joint angles (in radians).
    """
    default_pose = {}
    default_pose["left_thumb1"] = np.array([math.radians(-10), math.radians(10), math.radians(-10)], dtype=np.float32)
    default_pose["left_thumb2"] = np.array([math.radians(0), math.radians(10), math.radians(0)], dtype=np.float32)
    default_pose["left_thumb3"] = np.array([math.radians(0), math.radians(10), math.radians(0)], dtype=np.float32)
    default_pose["left_index1"] = np.array([math.radians(0), math.radians(0), math.radians(-0)], dtype=np.float32)
    default_pose["left_index2"] = np.array([math.radians(0), math.radians(0), math.radians(-0)], dtype=np.float32)
    default_pose["left_index3"] = np.array([math.radians(0), math.radians(0), math.radians(-0)], dtype=np.float32)
    default_pose["left_middle1"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_middle2"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_middle3"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_ring1"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_ring2"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_ring3"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_pinky1"] = np.array([math.radians(0), math.radians(-30), math.radians(-10)], dtype=np.float32)
    default_pose["left_pinky2"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)
    default_pose["left_pinky3"] = np.array([math.radians(0), math.radians(0), math.radians(-10)], dtype=np.float32)

    right_inverse = np.asarray([1, -1, -1])
    for k in list(default_pose.keys()):
        k1 = k.replace("left_", "right_")
        default_pose[k1] = default_pose[k] * right_inverse

    return default_pose


def get_flat_pose() -> dict:
    """
    Get the default joint angles for a flat hand pose for both left and right hands.

    Returns:
        dict: Mapping from bone names to numpy arrays of joint angles (in radians).
    """
    default_pose = {}
    default_pose["left_thumb1"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_thumb2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_thumb3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_index1"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_index2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_index3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_middle1"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_middle2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_middle3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_ring1"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_ring2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_ring3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_pinky1"] = np.array([math.radians(0), math.radians(-30), math.radians(0)], dtype=np.float32)
    default_pose["left_pinky2"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)
    default_pose["left_pinky3"] = np.array([math.radians(0), math.radians(0), math.radians(0)], dtype=np.float32)

    right_inverse = np.asarray([1, -1, -1])
    for k in list(default_pose.keys()):
        k1 = k.replace("left_", "right_")
        default_pose[k1] = default_pose[k] * right_inverse

    return default_pose



# fmt: on
