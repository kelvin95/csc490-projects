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
class PandasetConfig:
    """Configuration class for the Pandaset dataset.

    Attributes:
        basepath: Base directory to Pandaset download.
        sequence_ids: Sequence IDs of interest. None to use all.
        classes_to_keep: Label classes of interest. None to use all.
    """

    basepath: str
    sequence_ids: Optional[Sequence[int]] = None
    classes_to_keep: Optional[Sequence[LabelClass]] = (LabelClass.CAR,)


@dataclass
class PandasetOutput:
    """Output class for the Pandaset dataset in ego-centric coordinates.

    Attributes:
        lidar: Lidar pointcloud in vehicle frame as a [x, y, z] tensor
        sequence_id: Pandaset sequence id
        frame_id: Pandaset frame id
    """

    lidar: torch.Tensor
    labels: Dict[LabelClass, LabelData]
    sequence_id: int
    frame_id: int


class Pandaset(torch.utils.data.Dataset):
    """Dataset class for interacting with Pandaset."""

    def __init__(self, config: PandasetConfig):
        self._basepath = Path(config.basepath)
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

    def __getitem__(self, data_id: int) -> PandasetOutput:
        """Builds PandasetOutput from dataset entry id"""
        # Retrieve sequence and frame ids
        sequence_id, frame_id = self._samples[data_id]

        # Load vehicle pose
        rfu_vehicle_to_world = self._load_poses(sequence_id)[frame_id]
        lfu_vehicle_to_world = self._lfu_to_rfu.compose(rfu_vehicle_to_world)
        world_to_vehicle_lfu = lfu_vehicle_to_world.inverse()

        # Load lidar
        lidar_xyz_veh = self._load_lidar(sequence_id, frame_id, world_to_vehicle_lfu)

        # Load labels
        labels = self._load_labels(sequence_id, frame_id, world_to_vehicle_lfu)

        return PandasetOutput(
            lidar=lidar_xyz_veh,
            labels=labels,
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
                for lidar_file in sorted(lidar_files):
                    frame_id = int(lidar_file.name.split(".")[0])
                    yield (sequence_id, frame_id)

    def __len__(self) -> int:
        """Number of samples in dataset"""
        return len(self._samples)

    def _load_lidar(
        self, sequence_id: int, frame_id: int, world_to_vehicle: Transform3d
    ) -> torch.Tensor:
        """Loads the lidar pointcloud for (`sequence_id`, `frame_id`) and converts to vehicle frame.

        Args:
            sequence_id: Pandaset sequence id
            frame_id: Pandaset frame id
            world_to_vehicle: Rigid transform between world and ego pose

        Returns:
            Lidar pointcloud as a [x, y, z] tensor
        """
        lidar_path = (
            self._basepath / f"{sequence_id:03d}" / "lidar" / f"{frame_id:02d}.pkl.gz"
        )
        with gz.open(lidar_path, "rb") as f:
            lidar = pkl.load(f)

        # Only keep points from PANDAR64 lidar
        lidar_pandar64 = lidar.query("`d` == 0")

        # Convert to tensor
        lidar_xyz_world = torch.from_numpy(
            lidar_pandar64[["x", "y", "z"]].to_numpy(np.float32),
        )

        # Convert to vehicle frame
        lidar_xyz_veh = world_to_vehicle.transform_points(lidar_xyz_world)
        return lidar_xyz_veh

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
