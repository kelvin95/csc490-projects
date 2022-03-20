from dataclasses import dataclass

import torch


@dataclass
class Trajectories:
    """Dataclass for 2D bounding box detections.

    Args:
        centroids: [N x T x 2] centroids tensor. Each row is (x, y).
        yaws: [N x T] rotations in radians tensor.
        boxes: [N x 2] boxes tensor. Each row is (x_size, y_size).
    """

    centroids: torch.Tensor
    yaws: torch.Tensor
    boxes: torch.Tensor

    @property
    def centroids_x(self) -> torch.Tensor:
        """Return the x-axis centroid coordinates."""
        return self.centroids[:, :, 0]

    @property
    def centroids_y(self) -> torch.Tensor:
        """Return the y-axis centroid coordinates."""
        return self.centroids[:, :, 1]

    @property
    def boxes_x(self) -> torch.Tensor:
        """Return the x-axis bounding box size."""
        return self.expanded_boxes[:, :, 0]

    @property
    def boxes_y(self) -> torch.Tensor:
        """Return the y-axis bounding box size."""
        return self.expanded_boxes[:, :, 1]

    @property
    def flattened_centroids(self) -> torch.Tensor:
        return self.centroids.reshape(-1, 2)

    @property
    def flattened_yaws(self) -> torch.Tensor:
        return self.yaws.reshape(-1)

    @property
    def expanded_boxes(self) -> torch.Tensor:
        return self.boxes.unsqueeze(1).expand(self.centroids.shape)

    def to(self, device: torch.device) -> "Trajectories":
        """Return a copy of the detections moved to another device."""
        return Trajectories(
            self.centroids.to(device),
            self.yaws.to(device),
            self.boxes.to(device),
        )

    def __len__(self) -> int:
        """Return the number of actors."""
        return len(self.centroids)
