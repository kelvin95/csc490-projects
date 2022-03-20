import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from detection.modules.voxelizer import VoxelizerConfig
from detection.pandaset.dataset import PandasetConfig
from detection.pandaset.util import LabelClass
from tracking.dataset import OfflineTrackingDataset
from tracking.metrics.evaluator import Evaluator
from tracking.tracker import Tracker
from tracking.types import AssociateMethod, Tracklets
from tracking.visualization import plot_tracklets


def track(
    dataset_path,
    detection_path="tracking/detection_results/csc490_detector",
    result_path="tracking/tracking_results/results.pkl",
    tracker_associate_method=AssociateMethod.HUNGARIAN,
):
    print(f"Loading Pandaset from {dataset_path}")
    print(f"Loading dumped detection results from {detection_path}")
    voxelizer_config = VoxelizerConfig(
        x_range=(-76.0, 76.0),
        y_range=(-50.0, 50.0),
        z_range=(0.0, 10.0),
        step=0.25,
    )
    tracking_dataset = OfflineTrackingDataset(
        PandasetConfig(
            basepath=dataset_path,
            classes_to_keep=[LabelClass.CAR],
        ),
        detection_path,
        voxelizer_config,
    )
    print(
        f"Tracking with association method {AssociateMethod(tracker_associate_method)}"
    )
    tracking_results = {}
    for tracking_data in tqdm(tracking_dataset):
        seq_id = tracking_data.sequence_id
        tracking_inputs = tracking_data.tracking_inputs
        tracking_label = tracking_data.tracking_labels
        tracker = Tracker(
            track_steps=80, associate_method=AssociateMethod(tracker_associate_method)
        )
        tracker.track(tracking_inputs.bboxes, tracking_inputs.scores)
        tracking_pred = Tracklets(tracker.tracks)
        save_dict = {
            "sequence_id": seq_id,
            "tracking_label": tracking_label,
            "tracking_pred": tracking_pred,
        }
        tracking_results[seq_id] = save_dict

    print(f"Saving tracking results to {result_path}")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "wb") as f:
        pickle.dump(tracking_results, f)


def visualize(result_path="tracking/tracking_results/results.pkl"):
    viz_path = os.path.join(os.path.dirname(result_path), "viz")
    os.makedirs(viz_path, exist_ok=True)

    with open(result_path, "rb") as f:
        results_dict = pickle.load(f)

    for seq_id, result_dict in tqdm(results_dict.items()):
        tracking_label = result_dict["tracking_label"]
        tracking_pred = result_dict["tracking_pred"]
        fig, _ = plot_tracklets(
            tracking_pred,
            title=f"Estimated Tracklets for Pandaset Log{seq_id:03d} in World Frame",
        )
        fig.savefig(
            os.path.join(viz_path, f"log{seq_id:03d}_track_est.png"),
        )
        fig, _ = plot_tracklets(
            tracking_label,
            title=f"Ground-Truth Tracklets for Pandaset Log{seq_id:03d} in World Frame",
        )
        fig.savefig(os.path.join(viz_path, f"log{seq_id:03d}_track_gt.png"))
        plt.close("all")


def evaluate(result_path="tracking/tracking_results/results.pkl"):
    with open(result_path, "rb") as f:
        results_dict = pickle.load(f)

    evaluator = Evaluator()
    for seq_id, result_dict in tqdm(results_dict.items()):
        tracking_label = result_dict["tracking_label"]
        tracking_pred = result_dict["tracking_pred"]
        eval_results = evaluator.evaluate(tracking_label, tracking_pred)
        print(f"[Sequence: {seq_id:03d}]", eval_results)

    final_results_mean = evaluator.aggregate("mean")
    final_results_median = evaluator.aggregate("median")
    print(f"[Results (mean)", final_results_mean)
    print(f"[Results (median)", final_results_median)


if __name__ == "__main__":
    import fire

    fire.Fire()
