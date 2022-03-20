import numpy as np
import torch

from tracking.metrics.matching import Matching
from tracking.types import SingleTracklet, Tracklets


def test_same_gt_det():
    """Test when the detected tracklet is the same as GT tracklet"""
    tracklets = Tracklets(
        {
            "a": SingleTracklet(
                frame_ids=[0, 1],
                bboxes_traj=[
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 1.0],
            ),
            "b": SingleTracklet(
                frame_ids=[1, 2],
                bboxes_traj=[
                    torch.tensor([10.0, -4.0, 3.0, 2.0, 0.0]),
                    torch.tensor([10.0, -3.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 0.8],
            ),
        }
    )
    matching = Matching()
    matching.establish_correspondences(tracklets, tracklets)

    assert matching.frame_ids == [0, 1, 2]
    assert matching.matchings_list == [{"a": "a"}, {"a": "a", "b": "b"}, {"b": "b"}]
    assert matching.num_mismatches_list == [0, 0, 0]
    assert matching.num_matches_list == [1, 2, 1]
    assert matching.matched_dists_list == [[1.0], [1.0, 1.0], [1.0]]
    assert matching.num_false_positives_list == [0, 0, 0]
    assert matching.num_misses_list == [0, 0, 0]

    assert matching.compute_motp() == 1.0
    assert matching.compute_mota() == 1.0
    assert matching.compute_det_tracked_metrics(tracklets) == (1.0, 0.0, 0.0)


def test_diff_gt_det():
    """Simple test when the detected tracklet is different from GT tracklet"""
    tracklets_gt = Tracklets(
        {
            "a": SingleTracklet(
                frame_ids=[0, 1],
                bboxes_traj=[
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 1.0],
            ),
            "b": SingleTracklet(
                frame_ids=[1, 2],
                bboxes_traj=[
                    torch.tensor([10.0, -4.0, 3.0, 2.0, 0.0]),
                    torch.tensor([10.0, -3.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 0.8],
            ),
            "c": SingleTracklet(
                frame_ids=[2],
                bboxes_traj=[torch.tensor([100.0, -4.0, 3.0, 2.0, 0.0])],
                scores=[0.0],
            ),
        }
    )
    tracklets_det = Tracklets(
        {
            "a": SingleTracklet(
                frame_ids=[0, 1],
                bboxes_traj=[
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                    torch.tensor([1.0, 0.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 1.0],
            ),
            "b": SingleTracklet(
                frame_ids=[1, 2],
                bboxes_traj=[
                    torch.tensor([10.0, -4.0, 3.0, 2.0, 0.0]),
                    torch.tensor([20.0, -3.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 0.8],
            ),
            "c": SingleTracklet(
                frame_ids=[2, 3],
                bboxes_traj=[
                    torch.tensor([10.0, -3.0, 3.0, 2.0, 0.0]),
                    torch.tensor([10.0, -3.0, 3.0, 2.0, 0.0]),
                ],
                scores=[0.0, 0.8],
            ),
        }
    )
    matching = Matching()
    matching.establish_correspondences(tracklets_gt, tracklets_det)

    assert matching.frame_ids == [0, 1, 2, 3]
    assert matching.matchings_list == [{"a": "a"}, {"a": "a", "b": "b"}, {"b": "c"}, {}]
    assert matching.num_mismatches_list == [
        0,
        0,
        1,
        0,
    ]  # the mismatch happened with the b->c switch
    assert matching.num_matches_list == [1, 2, 1, 0]
    assert matching.matched_dists_list == [[1.0], [1.0, 1.0], [1.0], []]
    assert matching.num_false_positives_list == [0, 0, 1, 1]
    assert matching.num_misses_list == [0, 0, 1, 0]

    assert matching.compute_motp() == 1.0
    np.testing.assert_allclose(matching.compute_mota(), 0.2)
    np.testing.assert_allclose(
        np.array(matching.compute_det_tracked_metrics(tracklets_det)),
        np.array((1 / 3, 0.0, 2 / 3)),
    )
