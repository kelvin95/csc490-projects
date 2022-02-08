from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from detection.metrics.average_precision import (
    AveragePrecisionMetric,
    compute_average_precision,
)
from detection.metrics.types import EvaluationFrame
from detection.types import Detections


class EvaluationResult:
    """Dataclass to store the results of an evaluation."""

    def __init__(self, ap_metrics: Dict[float, AveragePrecisionMetric]) -> None:
        self._ap_metrics = ap_metrics

    @property
    def mean_ap(self) -> float:
        """Return the mean average precision over all thresholds."""
        if len(self._ap_metrics) == 0:
            return 0.0
        return np.mean(np.array([m.ap for m in self._ap_metrics.values()]))

    @property
    def ap_metrics(self) -> Dict[float, AveragePrecisionMetric]:
        """Return average precision metrics broken down by threshold."""
        return self._ap_metrics

    def as_dataframe(self) -> pd.DataFrame:
        """Return average precision as a data frame."""
        ap_dict = {th: [m.ap] for th, m in self._ap_metrics.items()}
        ap_dict["mean"] = [self.mean_ap]
        return pd.DataFrame.from_dict(ap_dict)

    def visualize(
        self, figsize: Tuple[int, int] = (8, 16), dpi: int = 75
    ) -> Optional[Tuple[Figure, Axes]]:
        """Visualize the evaluation results in matplotlib.

        Returns:
            Matplotlib figure and axis. `fig.show()` will display result.
        """
        if len(self._ap_metrics) == 0:
            return None

        fig, axes = plt.subplots(len(self._ap_metrics), figsize=figsize, dpi=dpi)
        for index, threshold in enumerate(self._ap_metrics.keys()):
            metric = self._ap_metrics[threshold]
            axes[index].plot(
                metric.pr_curve.recall.cpu().numpy(),
                metric.pr_curve.precision.cpu().numpy(),
            )
            axes[index].set_title(
                f"AP = {metric.ap:.2f} (threshold = {threshold:.1f}m)"
            )
        fig.supxlabel("Recall")
        fig.supylabel("Precision")
        fig.suptitle("Precision-Recall Curves")
        return fig, axes


class Evaluator:
    """Evaluates detections against ground truth labels."""

    def __init__(self, ap_thresholds: List[float]) -> None:
        """Initialization.

        Args:
            ap_thresholds: The thresholds used to evaluate AP.
        """
        self._ap_thresholds = ap_thresholds
        self._evaluation_frames: List[EvaluationFrame] = []

    def append(self, detections: Detections, labels: Detections) -> None:
        """Buffer a frame of detections and labels into the evaluator."""
        self._evaluation_frames.append(
            EvaluationFrame(
                detections.to(torch.device("cpu")),
                labels.to(torch.device("cpu")),
            )
        )

    def evaluate(self) -> EvaluationResult:
        """Evaluate the buffered frames and return results."""
        ap_metrics = {}
        for threshold in self._ap_thresholds:
            metric = compute_average_precision(self._evaluation_frames, threshold)
            ap_metrics[threshold] = metric
        return EvaluationResult(ap_metrics)

    def reset(self) -> None:
        """Reset the buffer."""
        self._evaluation_frames = []

    def __len__(self) -> int:
        """Return the size of the buffer."""
        return len(self._evaluation_frames)
