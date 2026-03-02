# Auto-generated skeleton by inspect_lerobot_dataset.py
# Edit as needed, then pass this file to --modality-config-path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.model.transforms import EmbodimentTag

pickmushroom_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=['ego_view', 'left_view', 'right_view'],
    ),

    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=['proprio.joint_pos', 'proprio.gripper_pos'],
        sin_cos_embedding_keys=['proprio.joint_pos'],
    ),

    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=['action.ee_delta', 'action.gripper_pos'],
        action_configs=[
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=['human.action.task_description'],
    ),
}

register_modality_config(pickmushroom_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

