from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from tracking.cost import iou_2d
from tracking.tracker import ActorID
from tracking.types import Tracklets


def order_tracklets_by_frame_ids(tracklets: Tracklets):
    """Reorganize the tracklets by frame ids"""
    frame_ids: List[int] = []
    # mapping between actor id and a sub-level dict that maps frame id to bbox
    tracklets_dict: Dict[ActorID, Dict[int, Tensor]] = {}
    for actor_id, tracklet in tracklets.tracks.items():
        tracklet_dict = {}
        for frame_id, bbox in zip(tracklet.frame_ids, tracklet.bboxes_traj):
            if frame_id not in frame_ids:
                frame_ids.append(frame_id)
            tracklet_dict[frame_id] = bbox
        tracklets_dict[actor_id] = tracklet_dict

    frame_ids.sort()
    # mapping between frame id and the actor ids present in the associated frame
    actors_by_frame_ids = {frame_id: [] for frame_id in frame_ids}
    for actor_id, tracklet in tracklets.tracks.items():
        for frame_id in tracklet.frame_ids:
            actors_by_frame_ids[frame_id].append(actor_id)
    return frame_ids, tracklets_dict, actors_by_frame_ids


class Matching:
    """Matching between M ground-truth tracklets (objects) and N estimated tracklets (hypothesis)"""

    def __init__(self, dist_th: float = 0.0):
        self.dist_th = (
            dist_th  # minimum distance/iou threshold for valid correspondences
        )

        self.frame_ids: List[int] = []  # list of frame ids at each time step
        # matched {object actor id: hypothesis actor id} at each time step
        self.matchings_list: List[Dict[ActorID, ActorID]] = []
        self.matched_dists_list: List[
            List[float]
        ] = []  # distance/iou of each matched object and hypothesis
        self.num_mismatches_list: List[
            int
        ] = []  # number of mismatches/switches at each time step
        self.num_matches_list: List[int] = []  # number of matches at each time step
        self.num_false_positives_list: List[
            int
        ] = []  # number of false positives at each time step
        self.num_misses_list: List[int] = []  # number of misses at each time step

    def reset(self):
        self.frame_ids = []
        self.matchings_list = []
        self.matched_dists_list = []
        self.num_mismatches_list = []
        self.num_matches_list = []
        self.num_false_positives_list = []
        self.num_misses_list = []

    def establish_correspondences(
        self, gt_tracklets: Tracklets, det_tracklets: Tracklets
    ):
        """Establish the correpondence between observed GT object and estimated hypothesis for each frame"""
        (
            gt_frame_ids,
            gt_tracklets_dict,
            gt_actors_by_frame_ids,
        ) = order_tracklets_by_frame_ids(gt_tracklets)
        (
            det_frame_ids,
            det_tracklets_dict,
            det_actors_by_frame_ids,
        ) = order_tracklets_by_frame_ids(det_tracklets)

        frame_ids = list(set(gt_frame_ids + det_frame_ids))
        frame_ids.sort()

        for idx, frame_id in enumerate(frame_ids):
            matchings = {}  # matchings for the current frame
            matched_dists = []  # bbox iou for matchings in the current frame

            gt_exist_in_cur_frame = frame_id in gt_actors_by_frame_ids
            det_exist_in_cur_frame = frame_id in det_actors_by_frame_ids

            # check if existing correspondences are still valid
            if gt_exist_in_cur_frame and det_exist_in_cur_frame:
                if idx > 0:
                    for o_id, h_id in self.matchings_list[-1].items():
                        if (
                            o_id in gt_actors_by_frame_ids[frame_id]
                            and h_id in det_actors_by_frame_ids[frame_id]
                        ):
                            o_bbox = gt_tracklets_dict[o_id][frame_id]
                            h_bbox = det_tracklets_dict[h_id][frame_id]
                            dist = iou_2d(
                                o_bbox.unsqueeze(0).cpu().numpy(),
                                h_bbox.unsqueeze(0).cpu().numpy(),
                            ).item()
                            if dist > self.dist_th:
                                # valid correspondence
                                matchings[o_id] = h_id
                                matched_dists.append(dist)

                # match remaining actors
                remaining_objects = [
                    o_id
                    for o_id in gt_actors_by_frame_ids[frame_id]
                    if o_id not in matchings.keys()
                ]
                remaining_hypotheses = [
                    h_id
                    for h_id in det_actors_by_frame_ids[frame_id]
                    if h_id not in matchings.values()
                ]
                if len(remaining_objects) > 0 and len(remaining_hypotheses) > 0:
                    remaining_o_bbox = np.stack(
                        [
                            gt_tracklets_dict[o_id][frame_id].cpu().numpy()
                            for o_id in remaining_objects
                        ]
                    )
                    remaining_h_bbox = np.stack(
                        [
                            det_tracklets_dict[h_id][frame_id].cpu().numpy()
                            for h_id in remaining_hypotheses
                        ]
                    )
                    cost_matrix = iou_2d(remaining_o_bbox, remaining_h_bbox)
                    cost_matrix[
                        cost_matrix <= self.dist_th
                    ] -= 1000000000  # invalid correspondences
                    row_inds, col_inds = linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    for i, j in zip(row_inds, col_inds):
                        if cost_matrix[i, j] > self.dist_th:
                            matchings[remaining_objects[i]] = remaining_hypotheses[j]
                            matched_dists.append(cost_matrix[i, j])

            num_mismatches = 0
            if gt_exist_in_cur_frame:
                if idx > 0:
                    for o_id in gt_actors_by_frame_ids[frame_id]:
                        if o_id in self.matchings_list[-1]:
                            if (
                                o_id in matchings
                                and self.matchings_list[-1][o_id] != matchings[o_id]
                            ):
                                num_mismatches += 1
            num_matches = len(matchings)
            if gt_exist_in_cur_frame:
                num_misses = len(gt_actors_by_frame_ids[frame_id]) - num_matches
            else:
                num_misses = 0
            if det_exist_in_cur_frame:
                num_false_positives = (
                    len(det_actors_by_frame_ids[frame_id]) - num_matches
                )
            else:
                num_false_positives = 0

            self.frame_ids.append(frame_id)
            self.matchings_list.append(matchings)
            self.num_matches_list.append(num_matches)
            self.num_mismatches_list.append(num_mismatches)
            self.matched_dists_list.append(matched_dists)
            self.num_false_positives_list.append(num_false_positives)
            self.num_misses_list.append(num_misses)

    def compute_motp(self) -> float:
        """Multiple object tracking precision"""
        # TODO: Replace this stub code.
        return 0.0

    def compute_mota(self) -> float:
        """Multiple object tracking accuracy"""
        # TODO: Replace this stub code.
        return 0.0

    def compute_gt_coverage_percentage(
        self, det_tracklets: Tracklets
    ) -> Dict[ActorID, float]:
        """Compute trajectory coverage with GT over the detected lifespan"""
        gt_coverage_pct_dict: Dict[ActorID, float] = {}
        for det_actor_id, tracklet in det_tracklets.tracks.items():
            num_matched = 0
            for matchings in self.matchings_list:
                if det_actor_id in matchings.values():
                    num_matched += 1
            gt_coverage_pct_dict[det_actor_id] = num_matched / tracklet.num_steps
        return gt_coverage_pct_dict

    def compute_det_tracked_metrics(
        self, det_tracklets: Tracklets
    ) -> Tuple[float, float, float]:
        """Compute the mostly/least/partially tracked metrics"""
        gt_coverage_pct_dict = self.compute_gt_coverage_percentage(det_tracklets)
        coverages = np.array(list(gt_coverage_pct_dict.values()))
        mostly_tracked = np.mean(coverages >= 0.8)
        mostly_lost = np.mean(coverages <= 0.2)
        partially_tracked = 1.0 - mostly_tracked - mostly_lost
        return mostly_tracked, mostly_lost, partially_tracked
