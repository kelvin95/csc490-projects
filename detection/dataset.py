from typing import List, Tuple

import torch
from torch import Tensor

from detection.model import DetectionModelConfig
from detection.modules.loss_target import DetectionLossTargetBuilder
from detection.modules.voxelizer import Voxelizer
from detection.pandaset.dataset import Pandaset, PandasetConfig
from detection.pandaset.util import LabelClass
from detection.types import Detections

TRAINING_SEQUENCE_IDS = [
    24,
    23,
    15,
    12,
    46,
    41,
    13,
    14,
    40,
    47,
    38,
    30,
    8,
    37,
    1,
    39,
    6,
    42,
    45,
    20,
    27,
    18,
    11,
    16,
    29,
    44,
    43,
]


VALIDATION_SEQUENCE_IDS = [17, 28, 19, 21, 3, 4, 32, 35, 34, 33, 5, 2]


class PandasetDataset:
    """Training and evaluation dataset for object detection."""

    def __init__(
        self,
        data_root: str,
        model_config: DetectionModelConfig,
        test: bool = False,
    ) -> None:
        """Initialization.

        Args:
            data_root: The root directory of the Pandaset dataset.
            model_config: The detection model configuration.
            test: If true, use the validation dataset split. Otherwise, use the training split.
        """
        if test is False:
            sequence_ids = TRAINING_SEQUENCE_IDS
        else:
            sequence_ids = VALIDATION_SEQUENCE_IDS

        self._pandaset = Pandaset(PandasetConfig(data_root, sequence_ids))
        self._voxelizer = Voxelizer(model_config.voxelizer)
        self._target_builder = DetectionLossTargetBuilder(
            model_config.voxelizer.bev_size, model_config.loss
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Detections]:
        """Returns training/validation data for one frame.

        Returns:
            A tuple consisting of:
            1. A [D x H x W] tensor containing the bird's eye view voxel representation
                of the LiDAR point cloud. Each cell is 1 if it's occupied by a LiDAR point
                and 0 otherwise.
            2. A [7 x H x W] tensor containing the ground truth target tensor for training
                the `DetectionModel` using a `DetectionLossFunction`. The 7 channels are
                (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
            3. A set of ground truth detections in bird's eye view image space coordinates.
        """
        pandaset_output = self._pandaset[idx]
        lidar_bev = self._voxelizer([pandaset_output.lidar])[0]  # [D x H x W]

        labels = Detections(
            pandaset_output.labels[LabelClass.CAR].centroids[:, :2],
            pandaset_output.labels[LabelClass.CAR].yaws,
            pandaset_output.labels[LabelClass.CAR].boxes[:, :2],
        )
        labels = self._voxelizer.project_detections(labels)

        targets = self._target_builder.build_target_tensor(labels)
        return lidar_bev.float(), targets, labels

    def __len__(self) -> None:
        """Return the size of this dataset."""
        return len(self._pandaset)


def custom_collate(
    batch: List[Tuple[Tensor, Tensor, Detections]]
) -> Tuple[Tensor, Tensor, List[Detections]]:
    """Custom collate function for torch dataloader."""
    bev, gt_tensor, labels = list(zip(*batch))
    return torch.stack(bev), torch.stack(gt_tensor), labels
