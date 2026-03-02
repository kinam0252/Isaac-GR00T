# /workspace/gr00t/pickmushroom_config.py

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag


pickmushroom_config = {
    # 1) VIDEO: use all 3 views
    "video": ModalityConfig(
        delta_indices=[0],  # current frame only [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
        modality_keys=["wrist_view", "left_view", "right_view"],  # must match meta/modality.json [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
    ),

    # 2) STATE: joint pos + gripper pos (sin/cos on joint angles)
    "state": ModalityConfig(
        delta_indices=[0],  # current state [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
        modality_keys=["proprio.joint_pos", "proprio.gripper_pos"],  # match meta/modality.json [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
        sin_cos_embedding_keys=["proprio.joint_pos"],  # recommended for radian joint angles [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
    ),

    # 3) ACTION: 16-step horizon, and DON'T apply extra relative conversion
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step prediction horizon [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
        modality_keys=["action.ee_delta", "action.gripper_pos"],  # match meta/modality.json [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
        action_configs=[
            # ee_delta is already delta in the dataset; keep as-is (avoid processor abs->rel conversion)
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # gripper pos stored as absolute scalar
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],  # length must match modality_keys [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
    ),

    # 4) LANGUAGE: must match meta/modality.json annotation keys
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],  # match meta/modality.json['annotation'] [1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
    ),
}

# Register for NEW_EMBODIMENT (required for custom embodiments) [2](https://github.com/wengmister/franka_joystick_teleop)[1](https://github.com/SpesRobotics/lerobot-teleoperator-teleop)
register_modality_config(pickmushroom_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
