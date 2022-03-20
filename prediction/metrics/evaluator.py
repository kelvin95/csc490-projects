from typing import Dict, List, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame

from prediction.metrics.ade_metrics import (
    compute_ADE,
    compute_FDE,
    compute_per_frame_err,
)
from prediction.metrics.types import EvaluationFrame
from prediction.types import Trajectories


class Evaluator:
    "Evaluates detections against ground truth labels"
    metrics = {
        "ADE": compute_ADE,
        "FDE": compute_FDE,
    }
    visual = {
        "frame_wise_err": compute_per_frame_err,
    }

    def __init__(self) -> None:
        self._evaluation_frames: List[EvaluationFrame] = []

    def append(self, trajectories: Trajectories, labels: Trajectories):
        self._evaluation_frames.append(
            EvaluationFrame(
                trajectories.to(torch.device("cpu")),
                labels.to(torch.device("cpu")),
            )
        )

    def reset(self) -> None:
        """Reset the buffer."""
        self._evaluation_frames = []

    def eval_visualize(self, output_dir):
        assert len(self._evaluation_frames) > 0

        eval_dict: Dict[str, float] = {}
        for metric_name, metric_func in self.visual.items():
            temp_dict: Dict[str, float] = {}
            for frame in self._evaluation_frames:
                sub_total = temp_dict.get("{}/total".format(metric_name), 0.0)
                sub_total += metric_func(
                    frame.trajectories.centroids, frame.labels.centroids
                )
                temp_dict["{}/total".format(metric_name)] = sub_total

                sub_count = temp_dict.get("{}/count".format(metric_name), 0.0)
                sub_count += 1
                temp_dict["{}/count".format(metric_name)] = sub_count

            eval_dict[metric_name] = (
                temp_dict["{}/total".format(metric_name)]
                / temp_dict["{}/count".format(metric_name)]
            )

            plt.plot(eval_dict[metric_name])
            plt.xlabel("Timestep")
            plt.ylabel("Mean Error (m)")
            plt.savefig(f"{output_dir}/{metric_name}.png")
            plt.close("all")

    def evaluate(self) -> DataFrame:
        assert len(self._evaluation_frames) > 0

        eval_dict: Dict[str, float] = {}
        for metric_name, metric_func in self.metrics.items():
            temp_dict: Dict[str, float] = {
                "{}/total".format(metric_name): 0,
                "{}/count".format(metric_name): 1e-6,
            }
            for frame in self._evaluation_frames:
                sub_total = temp_dict.get("{}/total".format(metric_name), 0.0)
                sub_total += metric_func(
                    frame.trajectories.centroids, frame.labels.centroids
                )
                temp_dict["{}/total".format(metric_name)] = sub_total

                sub_count = temp_dict.get("{}/count".format(metric_name), 0.0)
                sub_count += 1
                temp_dict["{}/count".format(metric_name)] = sub_count

            eval_dict[metric_name] = temp_dict["{}/total".format(metric_name)] / (
                temp_dict["{}/count".format(metric_name)]
            )

        for key, val in eval_dict.items():
            eval_dict[key] = [val]

        return pd.DataFrame.from_dict(eval_dict)
