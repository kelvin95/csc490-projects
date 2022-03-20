from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from detection.dataset import TRAINING_SEQUENCE_IDS, VALIDATION_SEQUENCE_IDS
from detection.pandaset.util import LabelClass, LabelData
from prediction.pandaset.dataset import PandasetPred, PandasetPredConfig
from prediction.types import Trajectories


class PandasetPredDataset:
    """Training and evaluation dataset for object detection."""

    def __init__(
        self,
        data_root: str,
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

        self._pandaset_pred = PandasetPred(PandasetPredConfig(data_root, sequence_ids))

    def __getitem__(self, idx: int) -> Tuple[Trajectories, Trajectories]:
        """Returns training/validation data for one frame.

        Returns:
            A tuple consisting of:
            1. A set of ground truth trajectory history in bird's eye view image space coordinates.
            2. A set of ground truth future trajectory labels in bird's eye view image space coordinates.
        """
        pandaset_pred_output = self._pandaset_pred[idx]

        # uid_to_idx gives us a list of actors that are present in most recent timestep
        history, uid_to_idx = self._convert_output_to_trajectory(
            pandaset_pred_output.pred_history
        )
        history_tensor = self._convert_trajectory_to_tensor(history)

        labels, _ = self._convert_output_to_trajectory(
            pandaset_pred_output.pred_labels, uid_to_idx=uid_to_idx
        )
        labels_tensor = self._convert_trajectory_to_tensor(labels)

        labels_tensor = labels_tensor[
            :, :, :2
        ]  # We only care about the centroids for labels

        return history_tensor, labels_tensor, labels

    def _convert_output_to_trajectory(
        self,
        pred_labels: List[Dict[LabelClass, LabelData]],
        uid_to_idx: Optional[Dict[int, int]] = None,
    ) -> Tuple[Trajectories, Dict[int, int]]:
        num_timesteps = len(pred_labels)
        if uid_to_idx is None:
            # Get all uids only of most recent timestep
            # because we only want actors present in most recent timestep
            all_uids = set()
            for class_id, label_data in pred_labels[-1].items():
                all_uids.update(label_data.uids)

            # Map uids to index in Trajectories object
            uid_to_idx = dict(zip(all_uids, range(len(all_uids))))
        num_actors = len(uid_to_idx)

        # Initialize trajectories object as all NaNs
        trajectories = Trajectories(
            centroids=torch.empty(num_actors, num_timesteps, 2) * float("nan"),
            yaws=torch.empty(num_actors, num_timesteps) * float("nan"),
            boxes=torch.empty(num_actors, 2) * float("nan"),
        )

        for timestep, frame_labels in enumerate(pred_labels):
            for class_id, label_data in frame_labels.items():
                if class_id is not LabelClass.CAR:
                    continue
                for i, uid in enumerate(label_data.uids):
                    if uid in uid_to_idx:
                        actor_idx = uid_to_idx[uid]
                        # Fill in the information for the actor and timestep
                        trajectories.centroids[
                            actor_idx, timestep
                        ] = label_data.centroids[i, :2]
                        trajectories.yaws[actor_idx, timestep] = label_data.yaws[i]
                        trajectories.boxes[actor_idx] = label_data.boxes[i, :2]
        return trajectories, uid_to_idx

    def _convert_trajectory_to_tensor(self, trajectories: Trajectories) -> Tensor:
        """Convert trajectories object into a tensor

        Repeats the boxes tensor over the time dimension, even through the box size doesn't
        change over time, so the shapes match
        """
        num_timesteps = trajectories.centroids.shape[-2]
        expanded_boxes = trajectories.boxes[:, None, :].expand(
            -1, num_timesteps, -1
        )  # [N x T x 2]
        expanded_yaws = trajectories.yaws[..., None]  # [N x T x 1]
        traj_tensor = torch.cat(
            [trajectories.centroids, expanded_yaws, expanded_boxes],
            dim=-1,
        )
        return traj_tensor

    def __len__(self) -> None:
        """Return the size of this dataset."""
        return len(self._pandaset_pred)


def custom_collate(
    batch: List[Tuple[Tensor, Tensor, Trajectories]]
) -> Tuple[List[Tensor], List[Tensor], List[Trajectories]]:
    """Custom collate function for torch dataloader."""
    history_tensors, labels_tensors, labels = list(zip(*batch))
    return history_tensors, labels_tensors, labels
