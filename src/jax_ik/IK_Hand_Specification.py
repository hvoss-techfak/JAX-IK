import numpy as np

from IK_SMPLX_Statics import get_pointing_pose, get_shaping_pose, get_flat_pose
from IK_objectives_jax import InitPoseObj, BoneDirectionObjective


def getPoseMask(pose_dict, controlled_bones):
    ret = np.zeros((len(controlled_bones), 3))
    for i, bone in enumerate(controlled_bones):
        if bone in pose_dict.keys():
            ret[i] = np.ones((3,))
    return ret.flatten()


def getBoneFillArray(pose_dict, controlled_bones):
    ret = np.zeros((len(controlled_bones), 3))
    for i, bone in enumerate(controlled_bones):
        if bone in pose_dict.keys():
            ret[i] = pose_dict[bone].flatten()
    return ret.flatten()


# TODO: We still need to rename these parameters to be more descriptive
class HandSpecification:
    def __init__(
        self,
        is_pointing=False,
        is_shaping=False,
        is_flat=False,
        look_forward=False,
        look_45_up=False,
        look_45_down=False,
        look_up=False,
        look_down=False,
        look_45_x_downwards=False,
        look_45_x_upwards=False,
        look_x_inward=False,
        look_to_body=False,
        arm_down=False,
        arm_45_down=False,
        arm_flat=False,
        special_obj=None,
    ):
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

    def get_objectives(self, left_hand=True, controlled_bones=None, full_trajectory=False, last_position=True, weight=1.0):
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
