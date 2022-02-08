from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Rectangle

from detection.pandaset.dataset import PandasetOutput
from detection.pandaset.util import CLASS_COLORMAP


def plot_frame(
    data: PandasetOutput, figsize: Tuple[int, int] = (16, 8)
) -> Tuple[Figure, Axes]:
    """Plots a Pandaset frame.

    Args:
        data: Output from the Pandaset dataset

    Returns:
        Matplotlib figure and axis. `fig.show()` will display result.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=75)

    # Plot LiDAR
    ax.scatter(data.lidar[:, 0], data.lidar[:, 1], s=1)

    # Plot labels
    for label_class, labels in data.labels.items():
        color = tuple(map(lambda x: x / 255.0, CLASS_COLORMAP[label_class]))
        for ix in range(labels.rank):
            x, y = labels.centroids[ix, :2]
            yaw = labels.yaws[ix].item()
            length, width = labels.boxes[ix, :2]
            desc = label_class.value if ix == 0 else None

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
                    lw=4,
                    label=desc,
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

    # Set axis options
    ax.set_xlim([-100, 100])
    ax.set_xlabel("meters")
    ax.set_ylim([-50, 50])
    ax.set_ylabel("meters")
    title = "Pandaset Frame "
    title += f"(Sequence {data.sequence_id} - Frame {data.frame_id})"
    ax.set_title(title)
    ax.set_facecolor("black")
    ax.legend(loc="best")
    return fig, ax
