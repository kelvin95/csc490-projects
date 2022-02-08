from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Detections:
    """Dataclass for 2D bounding box detections.

    Args:
        centroids: [N x 2] centroids tensor. Each row is (x, y).
        yaws: [N] rotations in radians tensor.
        boxes: [N x 2] boxes tensor. Each row is (x_size, y_size).
        scores: [N] detection scores tensor. None if ground truth.
    """

    centroids: torch.Tensor
    yaws: torch.Tensor
    boxes: torch.Tensor
    scores: Optional[torch.Tensor] = None

    @property
    def centroids_x(self) -> torch.Tensor:
        """Return the x-axis centroid coordinates."""
        return self.centroids[:, 0]

    @property
    def centroids_y(self) -> torch.Tensor:
        """Return the y-axis centroid coordinates."""
        return self.centroids[:, 1]

    @property
    def boxes_x(self) -> torch.Tensor:
        """Return the x-axis bounding box size."""
        return self.boxes[:, 0]

    @property
    def boxes_y(self) -> torch.Tensor:
        """Return the y-axis bounding box size."""
        return self.boxes[:, 1]

    def to(self, device: torch.device) -> "Detections":
        """Return a copy of the detections moved to another device."""
        return Detections(
            self.centroids.to(device),
            self.yaws.to(device),
            self.boxes.to(device),
            self.scores.to(device) if self.scores is not None else None,
        )

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.centroids)
