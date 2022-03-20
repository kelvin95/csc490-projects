from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import Tensor, nn

from prediction.modules.loss_function import PredictionLossConfig
from prediction.types import Trajectories
from prediction.utils.reshape import flatten, unflatten_batch
from prediction.utils.transform import transform_using_actor_frame


@dataclass
class PredictionModelConfig:
    """Prediction model configuration."""

    loss: PredictionLossConfig = field(
        default_factory=lambda: PredictionLossConfig(
            l1_loss_weight=1.0,
        )
    )
    num_history_timesteps: int = 10  # Number of timesteps in the history
    num_label_timesteps: int = 10  # Number of timesteps to predict


class PredictionModel(nn.Module):
    """A basic object Prediction model."""

    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__()

        # TODO: Implement
        # self._encoder = FILL IN

        # TODO: Implement
        # self._decoder = FILL IN

    @staticmethod
    def _preprocess(x_batches: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Preprocess the inputs

        1. Flatten batch and actor dimensions
        2. Transform each actor's history so that its position at the latest timestep is (0, 0) with 0 rad of yaw
            (i.e. it is in actor frame)
        3. Pad nans with zero
        4. Remove the bounding box size from the inputs
        5. Flatten the time and feature dimensions

        Args:
            x_batches (List[Tensor]): List of length batch_size of [N x T x 5] trajectories

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - preprocessed input trajectories [batch_size * N x T * 3]
                - id of each actor's batch in the flattened list [batch_size * N]
                - original position and yaw of each actor at the latest timestep in SDV frame [batch_size * N, 3]
        """
        x, batch_ids = flatten(x_batches)  # [batch_size * N x T x 5]
        original_x_pose = torch.clone(x[:, -1, :3])

        # Move positions to actor frame
        transformed_positions = transform_using_actor_frame(
            x[..., :2], x[:, -1, :3], translate_to=True
        )
        x[..., :2] = transformed_positions
        # Move yaw to actor frame
        x[..., 2] = x[..., 2] - x[:, -1:, 2]

        # Pad nans
        x[x.isnan()] = 0

        # Remove box size
        x = x[..., :3]

        x = x.flatten(1, 2)  # [batch_size * N x T * 3]

        return x, batch_ids, original_x_pose

    @staticmethod
    def _postprocess(
        out: Tensor, batch_ids: Tensor, original_x_pose: Tensor
    ) -> List[Tensor]:
        """Postprocess predictions

        1. Unflatten time and position dimensions
        2. Transform predictions back into SDV frame
        3. Unflatten batch and actor dimension

        Args:
            out (Tensor): predicted input trajectories [batch_size * N x T * 2]
            batch_ids (Tensor): id of each actor's batch in the flattened list [batch_size * N]
            original_x_pose (Tensor): original position and yaw of each actor at the latest timestep in SDV frame
                [batch_size * N, 3]

        Returns:
            List[Tensor]: List of length batch_size of output predicted trajectories in SDV frame [N x T x 2]
        """
        num_actors = len(batch_ids)
        out = out.reshape(num_actors, -1, 2)  # [batch_size * N x T x 2]

        # Transform from actor frame, to make the prediction problem easier
        transformed_out = transform_using_actor_frame(
            out, original_x_pose, translate_to=False
        )

        # Translate so that latest timestep for each actor is the origin

        out_batches = unflatten_batch(transformed_out, batch_ids)
        return out_batches

    def forward(self, x_batches: List[Tensor]) -> List[Tensor]:
        """Perform a forward pass of the model's neural network.

        Args:
            x_batches: A [batch_size x N x T_in x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A [batch_size x N x T_out x 2] tensor, representing the future trajectory
                centroid outputs.
        """
        x, batch_ids, original_x_pose = self._preprocess(x_batches)
        out = self._decoder(self._encoder(x))
        out_batches = self._postprocess(out, batch_ids, original_x_pose)
        return out_batches

    @torch.no_grad()
    def inference(self, history: Tensor) -> Trajectories:
        """Predict a set of 2d future trajectory predictions from the detection history

        Args:
            history: A [batch_size x N x T x 5] tensor, representing the input history
                centroid, yaw and size in a bird's eye view voxel representation.

        Returns:
            A set of 2D future trajectory centroid predictions.
        """
        self.eval()
        pred = self.forward([history])[0]  # shape: B * N x T x 2
        num_timesteps, num_coords = pred.shape[-2:]

        # Add dummy values for yaws and boxes here because we will fill them in from the ground truth
        return Trajectories(
            pred,
            torch.zeros(pred.shape[0], num_timesteps),
            torch.ones(pred.shape[0], num_coords),
        )
