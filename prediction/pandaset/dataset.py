import gzip as gz
import json
import pickle as pkl
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.transforms import Transform3d

from detection.pandaset.util import (
    LabelClass,
    LabelData,
    label_dataframe_to_dict,
    pose_dict_to_transform,
)


@dataclass
class PandasetPredConfig:
    """Configuration class for the Pandaset dataset.

    Attributes:
        basepath: Base directory to Pandaset download.
        sequence_ids: Sequence IDs of interest. None to use all.
        classes_to_keep: Label classes of interest. None to use all.
        num_pred_history_sweeps: Number of detection frames to load from history
        stride_pred_history_sweeps: (1 / stride) frames to keep in history
        num_pred_label_sweeps: Number of detection frames to load of future trajectories
        stride_pred_label_sweeps: (1 / stride) frames to keep in prediction labels
    """

    basepath: str
    sequence_ids: Optional[Sequence[int]] = None
    classes_to_keep: Optional[Sequence[LabelClass]] = (LabelClass.CAR,)
    num_pred_history_sweeps: int = 10
    stride_pred_history_sweeps: int = 1
    num_pred_label_sweeps: int = 50
    stride_pred_label_sweeps: int = 5


@dataclass
class PandasetPredOutput:
    """Output class for the Pandaset dataset in ego-centric coordinates.

    Attributes:
        pred_history: List (over time) of dictionary mapping class id to a detection frame of history
        pred_labels: List (over time) of dictionary mapping class id to a detection frame of future
        sequence_id: Pandaset sequence id
        frame_id: Pandaset frame id
    """

    pred_history: List[Dict[LabelClass, LabelData]]
    pred_labels: List[Dict[LabelClass, LabelData]]
    sequence_id: int
    frame_id: int


class PandasetPred(torch.utils.data.Dataset):
    """Dataset class for interacting with Pandaset."""

    def __init__(self, config: PandasetPredConfig):
        self._basepath = Path(config.basepath)
        self._num_pred_history_sweeps = config.num_pred_history_sweeps
        self._stride_pred_history_sweeps = config.stride_pred_history_sweeps
        self._num_pred_label_sweeps = config.num_pred_label_sweeps
        self._stride_pred_label_sweeps = config.stride_pred_label_sweeps
        self._samples = sorted(self._index_sequences(config.sequence_ids))
        assert (
            len(self._samples) > 0
        ), "Dataset appears empty, check that you have the correct basepath set"
        self._classes_to_keep = [
            label_cls.value
            for label_cls in (
                config.classes_to_keep if config.classes_to_keep else LabelClass
            )
        ]

        # Pandaset's "world" frame is (mostly) sequence-relative: The translation component is defined
        # with respect to the first frame (i.e. the origin is the start of the sequence) but the rotation
        # component is defined with respect to UTM coordinates: x follows longitude (east: +),
        # y follows latitude (north: +), and z is up.
        #
        # Pandaset's "vehicle" frame uses a x=right, y=front, z=up coordinate convention, whereas we use
        # x=front, y=left, z=up. To keep things consistent, we embed our convention into Pandaset by rotating
        # their poses pi/2 (90 deg) counter-clockwise around Z.
        self._lfu_to_rfu = Transform3d().rotate_axis_angle(angle=90, axis="Z")

    def __getitem__(self, data_id: int) -> PandasetPredOutput:
        """Builds PandasetPredOutput from dataset entry id"""
        # Retrieve sequence and frame ids
        sequence_id, frame_id = self._samples[data_id]

        # Load vehicle pose
        rfu_vehicle_to_world = self._load_poses(sequence_id)[frame_id]
        lfu_vehicle_to_world = self._lfu_to_rfu.compose(rfu_vehicle_to_world)
        world_to_vehicle_lfu = lfu_vehicle_to_world.inverse()

        # Load prediction history
        pred_history = self._load_pred_history(
            sequence_id, frame_id, world_to_vehicle_lfu
        )

        # Load prediction labels
        pred_labels = self._load_pred_labels(
            sequence_id, frame_id, world_to_vehicle_lfu
        )

        return PandasetPredOutput(
            pred_history=pred_history,
            pred_labels=pred_labels,
            sequence_id=sequence_id,
            frame_id=frame_id,
        )

    def _index_sequences(
        self, sequence_ids: Optional[Sequence[int]] = None
    ) -> Iterable[Tuple[int, int]]:
        """Indexes sequences and frames in dataset path.

        Args:
            sequence_ids: Sequence IDs of interest. None to use all.
        """
        for f in self._basepath.iterdir():
            if f.is_dir():
                sequence_id = int(f.name)

                # skip if the dataset is configured to skip this sequence
                if sequence_ids is not None and sequence_id not in sequence_ids:
                    continue

                lidar_files = (f / "lidar").glob("*.pkl.gz")
                sorted_lidar_files = sorted(lidar_files)
                for lidar_file in sorted_lidar_files:
                    frame_id = int(lidar_file.name.split(".")[0])
                    # Make sure avoid frames with incomplete history or labels
                    if (
                        frame_id < self._num_pred_history_sweeps
                        or frame_id
                        >= len(sorted_lidar_files) - self._num_pred_label_sweeps
                    ):
                        continue
                    yield (sequence_id, frame_id)

    def __len__(self) -> int:
        """Number of samples in dataset"""
        return len(self._samples)

    def _load_labels(
        self, sequence_id: int, frame_id: int, world_to_vehicle: Transform3d
    ) -> Dict[LabelClass, LabelData]:
        """Loads the labels for (`sequence_id`, `frame_id`) and convert to vehicle frame LabelData.

        Args:
            sequence_id: Pandaset sequence id
            frame_id: Pandaset frame id
            world_to_vehicle: Rigid transform between world and ego pose

        Returns:
            Dictionary of label_class -> label_data
        """
        labels_path = (
            self._basepath
            / f"{sequence_id:03d}"
            / "annotations"
            / "cuboids"
            / f"{frame_id:02d}.pkl.gz"
        )
        with gz.open(labels_path, "rb") as f:
            labels = pkl.load(f)

        # Only keep labels for PANDAR64 lidar and for specified classes
        labels_pandar64 = labels.query(
            f"`cuboids.sensor_id` in [0, -1] and `label` in {self._classes_to_keep}"
        )

        # Convert dataframe to dict
        labels = label_dataframe_to_dict(labels_pandar64, world_to_vehicle)

        return dict(labels)

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

    def _load_pred_history(
        self, sequence_id: int, frame_id: int, world_to_vehicle_lfu: Transform3d
    ):
        """Loads a list of labels as history for the (`sequence_id`, `frame_id`)

        Converts to List[LabelData] in current vehicle frame."""
        pred_history = []
        # Iterate backwards from the present so that we always get the latest frame regardless of the stride
        for history_frame_id in range(
            frame_id,
            frame_id - self._num_pred_history_sweeps,
            -self._stride_pred_history_sweeps,
        ):
            history_frame = self._load_labels(
                sequence_id, history_frame_id, world_to_vehicle_lfu
            )
            pred_history.append(history_frame)
        # Reverse the list to get oldest to newest order
        return pred_history[::-1]

    def _load_pred_labels(
        self, sequence_id: int, frame_id: int, world_to_vehicle_lfu: Transform3d
    ):
        """Loads a list of labels as future trajectory labels for the (`sequence_id`, `frame_id`)

        Converts to List[LabelData] in current vehicle frame."""
        pred_labels = []
        # The first label should be `self._stride_pred_label_sweeps` into the future
        # so the time intervals between labels are consistent
        for label_frame_id in range(
            frame_id + self._stride_pred_label_sweeps,
            frame_id + self._num_pred_label_sweeps + 1,
            self._stride_pred_label_sweeps,
        ):
            label_frame = self._load_labels(
                sequence_id, label_frame_id, world_to_vehicle_lfu
            )
            pred_labels.append(label_frame)
        return pred_labels
