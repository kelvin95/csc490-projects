from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

from tracking.types import TrackingInputs, Tracklets


def plot_tracklets(
    tracklets: Tracklets, figsize: Tuple[int, int] = (12, 8), title: str = ""
) -> Tuple[Figure, Axes]:
    """Plots all tracklets in one Pandaset sequence.

    Args:
        tracklets: Tracklets to be visualized

    Returns:
        Matplotlib figure and axis. `fig.show()` will display result.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Plot tracking labels
    for actor_id, tracklet in tracklets.tracks.items():
        color = np.random.rand(
            3,
        )
        for idx, bbox in enumerate(tracklet.bboxes_traj):
            x, y = bbox[:2]
            desc = actor_id if idx == 0 else None

            ax.add_patch(
                Ellipse(
                    (x, y),
                    1,
                    1,
                    edgecolor=color,
                    facecolor="none",
                    lw=1,
                    # label=desc,
                )
            )
    # Set axis options
    ax.set_xlim([-10, 550])
    ax.set_ylim([-10, 400])
    ax.set_title(title)
    return fig, ax


def plot_tracking_inputs(
    tracking_inputs: TrackingInputs, figsize: Tuple[int, int] = (12, 8)
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Plot detection results (i.e., tracking inputs)
    for frame_id, bboxes in zip(tracking_inputs.frame_ids, tracking_inputs.bboxes):
        color = np.random.rand(
            3,
        )
        for bbox in bboxes:
            x, y = bbox[:2]
            length, width = bbox[2:4]
            yaw = bbox[4]

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
                    lw=0.5,
                )
            )

    # Set axis options
    ax.set_xlim([-10, 550])
    ax.set_ylim([-10, 400])
    title = "Pandaset World Frame"
    ax.set_title(title)
    return fig, ax
