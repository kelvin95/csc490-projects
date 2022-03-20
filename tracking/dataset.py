import gzip as gz
import json
import os
import pickle
import pickle as pkl
from dataclasses import dataclass
from functools import lru_cache
from math import pi
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from pytorch3d.transforms import Transform3d, matrix_to_euler_angles

from detection.modules.voxelizer import Voxelizer, VoxelizerConfig
from detection.pandaset.dataset import PandasetConfig
from detection.pandaset.util import (
    LabelClass,
    label_dataframe_to_dict,
    pose_dict_to_transform,
)
from detection.types import Detections
from tracking.types import TrackingInputs, Tracklets


@dataclass
class TrackingData:
    """Output class for the Pandaset dataset in ego-centric coordinates.

    Attributes:
        tracking_inputs: TrackingInputs
        tracking_labels: Tracklets
    """

    sequence_id: int
    tracking_inputs: TrackingInputs
    tracking_labels: Tracklets


class OfflineTrackingDataset(torch.utils.data.Dataset):
    """Dataset class for interacting with Pandaset at log level.
    Note that tracking is conducted in the world coordinate (default in Pandaset).
    """

    def __init__(
        self, config: PandasetConfig, det_path: str, voxelizer_config: VoxelizerConfig
    ):
        self._basepath = Path(config.basepath)
        self._det_basepath = Path(det_path)
        # similar to Pandaset but each example (or sequence) takes 80 frames
        self._samples = sorted(self._index_sequences(config.sequence_ids))
        self._voxelizer = Voxelizer(voxelizer_config)

        assert (
            len(self._samples) > 0
        ), "Detection results appear empty, check that you have the correct det_path set"
        self._classes_to_keep = [
            label_cls.value
            for label_cls in (
                config.classes_to_keep if config.classes_to_keep else LabelClass
            )
        ]

        self._flu_to_rfu = Transform3d().rotate_axis_angle(angle=90, axis="Z")

    def __getitem__(self, data_id: int) -> TrackingData:
        """Builds PandasetOutput from dataset entry id"""
        # Retrieve sequence and frame ids
        sequence_id = self._samples[data_id]

        # Load vehicle pose
        rfu_vehicle_to_world = self._load_poses(sequence_id)
        flu_vehicle_to_world = [
            self._flu_to_rfu.compose(pose) for pose in rfu_vehicle_to_world
        ]
        world_to_vehicle_flu = [pose.inverse() for pose in flu_vehicle_to_world]

        # Load labels
        tracking_labels = self._load_labels(sequence_id)
        tracking_inputs = self._load_detections(sequence_id, flu_vehicle_to_world)

        return TrackingData(
            sequence_id=sequence_id,
            tracking_inputs=tracking_inputs,
            tracking_labels=tracking_labels,
        )

    def _index_sequences(
        self, sequence_ids: Optional[Sequence[int]] = None
    ) -> Iterable[int]:
        """Indexes sequences and frames in dataset path.

        Args:
            sequence_ids: Sequence IDs of interest. None to use all.
        """
        for f in self._det_basepath.iterdir():
            if f.is_dir() and f.name != "viz":
                sequence_id = int(f.name)
                assert (
                    self._basepath / f"{sequence_id:03d}"
                ).exists(), "Pandaset appears empty, check that you have the correct basepath set"
                # skip if the dataset is configured to skip this sequence
                if sequence_ids is not None and sequence_id not in sequence_ids:
                    continue
                yield sequence_id

    def __len__(self) -> int:
        """Number of samples in dataset"""
        return len(self._samples)

    def _load_detections(
        self, sequence_id: int, veh_to_world: List[Transform3d]
    ) -> List[Detections]:
        """Loads the detetion results for `sequence_id` and convert to world frame TrackingInput.

        Args:
            sequence_id: Pandaset sequence id
            veh_to_world: Rigid transform between vehicle and world coordinates

        Returns:
            Dictionary of label_class -> label_data
        """
        seq_path = self._det_basepath / f"{sequence_id:03d}"
        frame_ids: List[int] = []
        seq_bboxes: List[torch.Tensor] = []
        seq_scores: Optional[torch.Tensor] = []
        for dets_path in sorted(seq_path.iterdir()):
            frame_id = int(os.path.basename(dets_path)[:2])
            with open(dets_path, "rb") as f:
                data = pickle.load(f)
                frame_ids.append(frame_id)
                # if data["scores"] is None:
                #     data["scores"] = torch.ones([data["bboxes"].shape[0]]).float()
                dets_veh = self._voxelizer.unproject_detections(
                    Detections(
                        centroids=torch.tensor(data["bboxes"][:, :2]),
                        yaws=torch.tensor(data["bboxes"][:, 4]),
                        boxes=torch.tensor(data["bboxes"][:, 2:4]),
                        scores=torch.tensor(data["scores"])
                        if data["scores"] is not None
                        else None,
                    )
                )
                labels_xy = dets_veh.centroids
                labels_z = torch.zeros_like(dets_veh.yaws[:, None])
                # Project centroids from world to vehicle
                labels_xyz_veh = torch.cat((labels_xy, labels_z), dim=-1)
                labels_xyz_world = veh_to_world[frame_id].transform_points(
                    labels_xyz_veh
                )

                world_to_veh = veh_to_world[frame_id].inverse().get_matrix()
                _, _, yaw_ego = matrix_to_euler_angles(world_to_veh[0, :3, :3], "XYZ")

                dets_world = Detections(
                    centroids=labels_xyz_world[:, :2],
                    yaws=dets_veh.yaws + yaw_ego,
                    boxes=dets_veh.boxes,
                    scores=dets_veh.scores,
                )
                dets_proj = self._voxelizer.project_detections(dets_world)
                new_bboxes = torch.cat(
                    [dets_proj.centroids, dets_proj.boxes, dets_proj.yaws[:, None]],
                    axis=-1,
                )
                new_scores = dets_proj.scores
                seq_bboxes.append(new_bboxes)
                if seq_scores is not None and new_scores is not None:
                    seq_scores.append(new_scores)
                else:
                    seq_scores = None

        return TrackingInputs(
            frame_ids=frame_ids, seq_bboxes=seq_bboxes, seq_scores=seq_scores
        )

    def _load_labels(self, sequence_id: int) -> Tracklets:
        """Loads the labels for `sequence_id` and convert to LabelData (in world coordinate).

        Args:
            sequence_id: Pandaset sequence id
            world_to_vehicle: Rigid transform between world and ego pose

        Returns:
            Dictionary of label_class -> label_data
        """
        seq_path = self._basepath / f"{sequence_id:03d}" / "annotations" / "cuboids"
        seq_labels = []
        frame_ids = []
        for labels_path in seq_path.iterdir():
            frame_id = int(os.path.basename(labels_path)[:2])
            with gz.open(labels_path, "rb") as f:
                labels = pkl.load(f)

            # Only keep labels for PANDAR64 lidar and for specified classes
            labels_pandar64 = labels.query(
                f"`cuboids.sensor_id` in [0, -1] and `label` in {self._classes_to_keep}"
            )

            # Convert dataframe to dict (stay at world coordinate)
            labels = label_dataframe_to_dict(labels_pandar64, Transform3d())
            frame_ids.append(frame_id)
            seq_labels.append(labels)

        tracking_labels = Tracklets.from_seq_labels(
            frame_ids, seq_labels, self._voxelizer
        )
        return tracking_labels

    @lru_cache
    def _load_poses(self, sequence_id: int) -> List[Transform3d]:
        """Loads all the poses for sequence with id sequence_id

        Args:
            sequence_id: Pandaset sequence id

        Returns:
            List of pose transforms for each frame in the sequence
        """
        path = self._basepath / f"{sequence_id:03d}" / "lidar" / "poses.json"
        with open(path, "r") as f:
            poses = json.load(f)
            return [pose_dict_to_transform(p) for p in poses]
