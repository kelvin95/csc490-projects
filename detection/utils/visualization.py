from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Rectangle

from detection.types import Detections


def plot_box(
    ax: Axes,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    color: Any,
    label: str,
) -> None:
    """Plot a bounding box onto the given axes."""
    dx = np.cos(yaw) * length - np.sin(yaw) * width
    dy = np.sin(yaw) * length + np.cos(yaw) * width

    # Plot rectangle
    ax.add_patch(
        Rectangle(
            (x - dx / 2, y - dy / 2),
            length,
            width,
            np.rad2deg(yaw),
            edgecolor=color,
            facecolor="none",
            lw=2,
            label=label,
        )
    )

    # Plot orientation arrow
    ax.add_patch(
        Arrow(
            x,
            y,
            np.cos(yaw) * length / 2,
            np.sin(yaw) * length / 2,
            edgecolor=color,
            facecolor=color,
            capstyle="projecting",
            lw=1,
        )
    )


def visualize_detections(
    pointcloud: torch.Tensor,
    detections: Detections,
    labels: Optional[Detections] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 75,
) -> Tuple[Figure, Axes]:
    """Plots a frame of detections and ground truth labels.

    Args:
        pointcloud: [N x 3] lidar point cloud. Each row is (x, y, z) coordinates.
        detections: The detections in the frame.
        labels: The ground truth labels in the frame.

    Returns:
        Matplotlib figure and axis. `fig.show()` will display result.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot LiDAR
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], s=1)

    # Plot labels
    if labels is not None:
        for ix in range(len(labels)):
            plot_box(
                ax,
                labels.centroids_x[ix].item(),
                labels.centroids_y[ix].item(),
                labels.yaws[ix].item(),
                labels.boxes_x[ix].item(),
                labels.boxes_y[ix].item(),
                (0.0, 1.0, 0.0, 0.5),
                "Ground Truth",
            )

    # Plot detections
    for ix in range(len(detections)):
        plot_box(
            ax,
            detections.centroids_x[ix].item(),
            detections.centroids_y[ix].item(),
            detections.yaws[ix].item(),
            detections.boxes_x[ix].item(),
            detections.boxes_y[ix].item(),
            (1.0, 0.0, 0.0, 0.5),
            "Detections",
        )

    return fig, ax
