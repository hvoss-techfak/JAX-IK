import numpy as np

from jax_ik.smplx_statics import get_pointing_pose, get_shaping_pose, get_flat_pose
from jax_ik.objectives import InitPoseObj, BoneDirectionObjective, SDFSelfCollisionPenaltyObj


def getPoseMask(pose_dict: dict, controlled_bones: list) -> np.ndarray:
    """
    Create a flat mask array indicating which bones in controlled_bones are present in pose_dict.

    Args:
        pose_dict (dict): Dictionary mapping bone names to pose arrays.
        controlled_bones (list): List of bone names being controlled.

    Returns:
        np.ndarray: Flattened mask array (len(controlled_bones) * 3,) with 1s for present bones, 0s otherwise.
    """
    ret = np.zeros((len(controlled_bones), 3))
    for i, bone in enumerate(controlled_bones):
        if bone in pose_dict.keys():
            ret[i] = np.ones((3,))
    return ret.flatten()


def getBoneFillArray(pose_dict: dict, controlled_bones: list) -> np.ndarray:
    """
    Create a flat array of target joint angles for all controlled bones, filling in zeros for missing bones.

    Args:
        pose_dict (dict): Dictionary mapping bone names to pose arrays.
        controlled_bones (list): List of bone names being controlled.

    Returns:
        np.ndarray: Flattened array (len(controlled_bones) * 3,) of joint angles.
    """
    ret = np.zeros((len(controlled_bones), 3))
    for i, bone in enumerate(controlled_bones):
        if bone in pose_dict.keys():
            ret[i] = pose_dict[bone].flatten()
    return ret.flatten()


# TODO: We still need to rename these parameters to be more descriptive
class HandSpecification:
    """
    Helper class to build a list of IK objectives for hand/arm gestures and orientations.

    Set the desired flags in the constructor, then call get_objectives() to get a list of objectives.
    """

    def __init__(
        self,
        is_pointing: bool = False,
        is_shaping: bool = False,
        is_flat: bool = False,
        look_forward: bool = False,
        look_45_up: bool = False,
        look_45_down: bool = False,
        look_up: bool = False,
        look_down: bool = False,
        look_45_x_downwards: bool = False,
        look_45_x_upwards: bool = False,
        look_x_inward: bool = False,
        look_to_body: bool = False,
        arm_down: bool = False,
        arm_45_down: bool = False,
        arm_flat: bool = False,
        special_obj=None,
        penalize_self_collision: bool = False
    ):
        """
        Initialize the hand specification with various pose and direction flags.

        Args:
            is_pointing (bool): Use pointing hand pose.
            is_shaping (bool): Use shaping hand pose.
            is_flat (bool): Use flat hand pose.
            look_forward (bool): Make wrist point forward.
            look_45_up (bool): Make wrist point 45 degrees up.
            look_45_down (bool): Make wrist point 45 degrees down.
            look_up (bool): Make wrist point up.
            look_down (bool): Make wrist point down.
            look_45_x_downwards (bool): Make wrist point diagonally down.
            look_45_x_upwards (bool): Make wrist point diagonally up.
            look_x_inward (bool): Make wrist point inward.
            look_to_body (bool): Make wrist point toward body.
            arm_down (bool): Make upper arm point down.
            arm_45_down (bool): Make upper arm point diagonally down.
            arm_flat (bool): Make upper arm point forward.
            special_obj: Any special custom objective.
            penalize_self_collision (bool): Add self-collision penalty for the hand.
        """
        self.is_pointing = is_pointing
        self.is_shaping = is_shaping
        self.is_flat = is_flat
        self.look_forward = look_forward
        self.look_45_up = look_45_up
        self.look_45_down = look_45_down
        self.look_up = look_up
        self.look_down = look_down
        self.look_x_45_in_downwards = look_45_x_downwards
        self.look_x_45_in_upwards = look_45_x_upwards
        self.look_x_inward = look_x_inward
        self.look_to_body = look_to_body
        self.arm_down = arm_down
        self.arm_45_down = arm_45_down
        self.arm_flat = arm_flat
        self.special_obj = special_obj
        self.penalize_self_collision = penalize_self_collision

    def get_objectives(
        self,
        left_hand: bool = True,
        controlled_bones: list = None,
        full_trajectory: bool = False,
        last_position: bool = True,
        weight: float = 1.0
    ) -> list:
        """
        Build a list of IK objectives for the specified hand/arm configuration.

        Args:
            left_hand (bool): If True, use left hand; otherwise, right hand.
            controlled_bones (list): List of bone names being controlled.
            full_trajectory (bool): If True, apply pose objectives to all frames.
            last_position (bool): If True, apply pose objectives to last frame.
            weight (float): Weight for pose objectives.

        Returns:
            list: List of ObjectiveFunction instances for this hand specification.
        """
        ret_objectives = []

        hand_str = "left" if left_hand else "right"
        hand_bones = controlled_bones
        x_direction = 1 if hand_str == "left" else -1
        obj = BoneDirectionObjective(
            bone_name=f"{hand_str}_elbow",
            use_head=False,
            directions=[[x_direction, 0, 0]],
            weight=0.01,
        )
        ret_objectives.append(obj)

        if self.penalize_self_collision:
            # Add self-collision penalty for the hand
            ret_objectives.append(
                SDFSelfCollisionPenaltyObj(
                    bone_names=[f"{hand_str}_wrist", f"{hand_str}_elbow"],
                    num_samples_per_bone=5,
                    min_dist=0.05,  # Ignore points closer than 5cm in rest pose
                    weight=0.1 * weight,  # Low weight to gently guide
                )
            )

        if self.is_pointing:
            bones = get_pointing_pose()
            bones_out = getBoneFillArray(bones, hand_bones)
            bones_mask = getPoseMask(bones, hand_bones)
            ret_objectives.append(
                InitPoseObj(
                    init_rot=bones_out,
                    full_trajectory=full_trajectory,
                    last_position=last_position,
                    weight=weight,
                    mask=bones_mask,
                )
            )
        if self.is_shaping:
            bones = get_shaping_pose()
            bones_out = getBoneFillArray(bones, hand_bones)
            bones_mask = getPoseMask(bones, hand_bones)
            ret_objectives.append(
                InitPoseObj(
                    init_rot=bones_out,
                    full_trajectory=full_trajectory,
                    last_position=last_position,
                    weight=weight,
                    mask=bones_mask,
                )
            )
        if self.is_flat:
            bones = get_flat_pose()
            bones_out = getBoneFillArray(bones, hand_bones)
            bones_mask = getPoseMask(bones, hand_bones)
            ret_objectives.append(
                InitPoseObj(
                    init_rot=bones_out,
                    full_trajectory=full_trajectory,
                    last_position=last_position,
                    weight=weight,
                    mask=bones_mask,
                )
            )
        if self.look_forward:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, 1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_index3",
                use_head=False,
                directions=[[0, 0, 1]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_45_up:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, 1, -1]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_45_down:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, 1, 1]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_up:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, -1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_down:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, 1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_x_45_in_downwards:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[x_direction, 1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_x_45_in_upwards:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[x_direction, -1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_x_inward:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[x_direction, 0, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.look_to_body:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_wrist",
                use_head=False,
                directions=[[0, 0, 1]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.arm_down:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_shoulder",
                use_head=False,
                directions=[[0, -1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.arm_45_down:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_shoulder",
                use_head=False,
                directions=[[x_direction, -1, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)
        if self.arm_flat:
            obj = BoneDirectionObjective(
                bone_name=f"{hand_str}_shoulder",
                use_head=False,
                directions=[[x_direction, 0, 0]],
                weight=weight,
            )
            ret_objectives.append(obj)

        return ret_objectives
