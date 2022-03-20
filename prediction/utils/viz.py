from turtle import color
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from detection.utils.visualization import plot_box
from prediction.types import Trajectories


def visualize_trajectories(
    trajectories: Trajectories,
    start_color: Tuple[float, float, float],
    end_color: Tuple[float, float, float],
    name: str,
    fig: Figure,
    ax: Axes,
) -> Tuple[Figure, Axes]:
    """Plots a frame of detections and ground truth labels.

    Args:
        trajectories: [N, T, 2] trajectories to display. Each final coordinate is the (x,y) at that timestep T
        color: color to display the trajectories in (R, G, B)

    Returns:
        Matplotlib figure and axis. `fig.show()` will display result.
    """

    # Plot trajectories
    centroids_x = trajectories.centroids_x
    centroids_y = trajectories.centroids_y
    yaws = trajectories.yaws
    boxes_x = trajectories.boxes_x
    boxes_y = trajectories.boxes_y

    colors = []
    for ix in range(centroids_x.shape[0]):
        for t in range(centroids_x.shape[1]):
            ratio = t / (centroids_x.shape[1] - 0.999)
            new_color = (
                start_color[0] * (1 - ratio) + end_color[0] * ratio,
                start_color[1] * (1 - ratio) + end_color[1] * ratio,
                start_color[2] * (1 - ratio) + end_color[2] * ratio,
                0.3,
            )
            plot_box(
                ax,
                centroids_x[ix, t].item(),
                centroids_y[ix, t].item(),
                yaws[ix, t].item(),
                boxes_x[ix, t].item(),
                boxes_y[ix, t].item(),
                new_color,
                name,
            )
            colors.append(new_color)

    axins1 = inset_axes(ax, width="10%", height="4%", loc="upper left")
    cmap = LinearSegmentedColormap.from_list(name, [start_color, end_color])
    norm = Normalize(vmin=0, vmax=trajectories.centroids_x.shape[1])
    cb1 = ColorbarBase(axins1, cmap=cmap, norm=norm, orientation="horizontal")
    cb1.set_label(name)

    return fig, ax


def vis_pred_labels(
    pred_trajectories: Trajectories,
    label_trajectories: Trajectories,
    figsize: Tuple[int, int] = (16, 16),
    dpi: int = 150,
):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
    start_pred_color = (1.0, 0.0, 0.0)
    end_pred_color = (0.0, 1.0, 1.0)

    start_label_color = (0.0, 0.0, 1.0)
    end_label_color = (1.0, 0.0, 1.0)

    visualize_trajectories(
        pred_trajectories,
        start_pred_color,
        end_pred_color,
        "Predictions Time",
        fig,
        ax1,
    )
    visualize_trajectories(
        label_trajectories,
        start_label_color,
        end_label_color,
        "Labels Time",
        fig,
        ax2,
    )

    ax1.set_xlim(
        [
            min(
                [
                    pred_trajectories.centroids_x.min() - 10,
                    label_trajectories.centroids_x[
                        ~label_trajectories.centroids_x.isnan()
                    ].min()
                    - 10,
                ]
            ),
            max(
                [
                    pred_trajectories.centroids_x.max() + 10,
                    label_trajectories.centroids_x[
                        ~label_trajectories.centroids_x.isnan()
                    ].max()
                    + 10,
                ]
            ),
        ]
    )
    ax2.set_xlim(
        [
            min(
                [
                    pred_trajectories.centroids_x.min() - 10,
                    label_trajectories.centroids_x[
                        ~label_trajectories.centroids_x.isnan()
                    ].min()
                    - 10,
                ]
            ),
            max(
                [
                    pred_trajectories.centroids_x.max() + 10,
                    label_trajectories.centroids_x[
                        ~label_trajectories.centroids_x.isnan()
                    ].max()
                    + 10,
                ]
            ),
        ]
    )
    ax1.set_ylim(
        [
            min(
                [
                    pred_trajectories.centroids_y.min() - 10,
                    label_trajectories.centroids_y[
                        ~label_trajectories.centroids_y.isnan()
                    ].min()
                    - 10,
                ]
            ),
            max(
                [
                    pred_trajectories.centroids_y.max() + 10,
                    label_trajectories.centroids_y[
                        ~label_trajectories.centroids_y.isnan()
                    ].max()
                    + 10,
                ]
            ),
        ]
    )
    ax2.set_ylim(
        [
            min(
                [
                    pred_trajectories.centroids_y.min() - 10,
                    label_trajectories.centroids_y[
                        ~label_trajectories.centroids_y.isnan()
                    ].min()
                    - 10,
                ]
            ),
            max(
                [
                    pred_trajectories.centroids_y.max() + 10,
                    label_trajectories.centroids_y[
                        ~label_trajectories.centroids_y.isnan()
                    ].max()
                    + 10,
                ]
            ),
        ]
    )

    return fig, ax1
