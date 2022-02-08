from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


def heatmap_weighted_mse_loss(
    targets: Tensor, predictions: Tensor, heatmap: Tensor, heatmap_threshold: float
) -> Tensor:
    """Compute the mean squared error (MSE) loss between `predictions` and `targets`, weighted by a heatmap.

    Specifically, the heatmap-weighted MSE loss can be computed as follows:
    1. Compute the MSE loss between `predictions` and `targets` along the C dimension.
        This should give us a [batch_size x H x W] tensor `mse_loss`.
    2. Compute a binary mask based on whether the values of `heatmap` exceeds `heatmap_threshold`.
        This should give us a [batch_size x H x W] tensor `mask`.
    3. Compute the mean of `mse_loss` weighted by `heatmap` and masked by `mask`.
        This gives us our final scalar loss.

    Args:
        targets: A [batch_size x C x H x W] tensor, containing the ground truth targets.
        predictions: A [batch_size x C x H x W] tensor, containing the predictions.
        heatmap: A [batch_size x 1 x H x W] tensor, representing the ground truth heatmap.
        heatmap_threshold: We compute MSE loss for only elements where `heatmap > heatmap_threshold`.

    Returns:
        A scalar MSE loss between `predictions` and `targets`, aggregated as a
            weighted average using the provided `heatmap`.
    """
    # TODO: Replace this stub code.
    return torch.sum(predictions) * 0.0


@dataclass
class DetectionLossConfig:
    """Detection loss function configuration.

    Attributes:
        heatmap_loss_weight: The multiplicative weight of the heatmap loss.
        offset_loss_weight: The multiplicative weight of the offset loss.
        size_loss_weight: The multiplicative weight of the size loss.
        heading_loss_weight: The multiplicative weight of the heading loss.
        heatmap_threshold: A scalar threshold that controls whether we ignore the loss
            at a given location. In particular, we ignore the loss for all locations
            where the ground truth heatmap has a value less than or equal to `heatmap_threshold`.
        heatmap_norm_scale: A scalar value that scales the spread of a heatmap.
            The larger the value, the smaller the spread of the heatmap.
            See `detection/modules/loss_target.py` for usage details.
    """

    heatmap_loss_weight: float
    offset_loss_weight: float
    size_loss_weight: float
    heading_loss_weight: float
    heatmap_threshold: float
    heatmap_norm_scale: float


@dataclass
class DetectionLossMetadata:
    """Detailed breakdown of the detection loss."""

    total_loss: torch.Tensor
    heatmap_loss: torch.Tensor
    offset_loss: torch.Tensor
    size_loss: torch.Tensor
    heading_loss: torch.Tensor


class DetectionLossFunction(torch.nn.Module):
    """A loss function to train a detection model."""

    def __init__(self, config: DetectionLossConfig) -> None:
        super(DetectionLossFunction, self).__init__()
        self._heatmap_loss_weight = config.heatmap_loss_weight
        self._offset_loss_weight = config.offset_loss_weight
        self._size_loss_weight = config.size_loss_weight
        self._heading_loss_weight = config.heading_loss_weight
        self._heatmap_threshold = config.heatmap_threshold

    def forward(
        self, predictions: Tensor, targets: Tensor
    ) -> Tuple[torch.Tensor, DetectionLossMetadata]:
        """Compute the loss between the predicted detections and target labels.

        Args:
            predictions: A [batch_size x 7 x H x W] tensor containing the outputs of `DetectionModel`.
                The 7 channels are (heatmap, offset_x, offset_y, length, width, sin_theta, cos_theta).
            targets: A [batch_size x 7 x H x W] tensor containing the ground truth output.
                The 7 channels are (heatmap, offset_x, offset_y, length, width, sin_theta, cos_theta).

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        # 1. Unpack the targets tensor.
        target_heatmap = targets[:, 0:1]  # [B x 1 x H x W]
        target_offsets = targets[:, 1:3]  # [B x 2 x H x W]
        target_sizes = targets[:, 3:5]  # [B x 2 x H x W]
        target_headings = targets[:, 5:7]  # [B x 2 x H x W]

        # 2. Unpack the predictions tensor.
        predicted_heatmap = torch.sigmoid(predictions[:, 0:1])  # [B x 1 x H x W]
        predicted_offsets = predictions[:, 1:3]  # [B x 2 x H x W]
        predicted_sizes = predictions[:, 3:5]  # [B x 2 x H x W]
        predicted_headings = predictions[:, 5:7]  # [B x 2 x H x W]

        # 3. Compute individual loss terms for heatmap, offset, size, and heading.
        heatmap_loss = ((target_heatmap - predicted_heatmap) ** 2).mean()
        offset_loss = heatmap_weighted_mse_loss(
            target_offsets, predicted_offsets, target_heatmap, self._heatmap_threshold
        )
        size_loss = heatmap_weighted_mse_loss(
            target_sizes, predicted_sizes, target_heatmap, self._heatmap_threshold
        )
        heading_loss = heatmap_weighted_mse_loss(
            target_headings, predicted_headings, target_heatmap, self._heatmap_threshold
        )

        # 4. Aggregate losses using the configured weights.
        total_loss = (
            heatmap_loss * self._heatmap_loss_weight
            + offset_loss * self._offset_loss_weight
            + size_loss * self._size_loss_weight
            + heading_loss * self._heading_loss_weight
        )

        loss_metadata = DetectionLossMetadata(
            total_loss, heatmap_loss, offset_loss, size_loss, heading_loss
        )
        return total_loss, loss_metadata
