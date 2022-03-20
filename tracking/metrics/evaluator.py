from dataclasses import dataclass
from typing import List

import numpy as np

from tracking.metrics.matching import Matching
from tracking.types import Tracklets


@dataclass
class EvaluationResult:
    mota: float
    motp: float
    mostly_tracked: float
    mostly_lost: float
    partially_tracked: float


class Evaluator:
    mota: List[float] = []
    motp: List[float] = []
    mostly_tracked: List[float] = []
    mostly_lost: List[float] = []
    partially_tracked: List[float] = []

    def __init__(self, matching_min_th: float = 0.0):
        self.matching = Matching(dist_th=matching_min_th)

    def evaluate(self, gt_tracklets: Tracklets, det_tracklets: Tracklets):
        self.matching.reset()
        self.matching.establish_correspondences(gt_tracklets, det_tracklets)
        (
            mostly_tracked,
            mostly_lost,
            partially_tracked,
        ) = self.matching.compute_det_tracked_metrics(det_tracklets)
        mota, motp = self.matching.compute_mota(), self.matching.compute_motp()
        self.collect(mota, motp, mostly_tracked, mostly_lost, partially_tracked)
        return EvaluationResult(
            mota=mota,
            motp=motp,
            mostly_tracked=mostly_tracked,
            mostly_lost=mostly_lost,
            partially_tracked=partially_tracked,
        )

    def collect(
        self,
        mota: float,
        motp: float,
        mostly_tracked: float,
        mostly_lost: float,
        partially_tracked: float,
    ):
        self.mota.append(mota)
        self.motp.append(motp)
        self.mostly_tracked.append(mostly_tracked)
        self.mostly_lost.append(mostly_lost)
        self.partially_tracked.append(partially_tracked)

    def aggregate(self, criterion="mean"):
        if criterion == "mean":
            agg_op = np.mean
        elif criterion == "median":
            agg_op = np.median
        else:
            raise NotImplementedError(f"Unsupported criterion option: {criterion}")
        return EvaluationResult(
            mota=agg_op(self.mota),
            motp=agg_op(self.motp),
            mostly_tracked=agg_op(self.mostly_tracked),
            mostly_lost=agg_op(self.mostly_lost),
            partially_tracked=agg_op(self.partially_tracked),
        )
