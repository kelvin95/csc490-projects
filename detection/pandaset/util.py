from dataclasses import dataclass
from enum import Enum, unique
from math import pi
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_euler_angles,
    quaternion_invert,
    quaternion_to_matrix,
)


@unique
class LabelClass(str, Enum):
    """Pandaset Label Classes"""

    CAR = "Car"
    PEDESTRIAN = "Pedestrian"
    PICKUP = "Pickup Truck"
    TEMPORARY_CONSTRUCTION_BARRIER = "Temporary Construction Barriers"
    PEDESTRIAN_WITH_OBJECT = "Pedestrian with Object"
    CONES = "Cones"
    SIGNS = "Signs"
    MEDIUM_SIZED_TRUCK = "Medium-sized Truck"
    PYLON = "Pylons"
    ROLLING_CONTAINER = "Rolling Containers"
    BICYCLE = "Bicycle"
    MOTORCYCLE = "Motorcycle"
    BUS = "Bus"
    ROAD_BARRIER = "Road Barriers"
    CONSTRUCTION_SIGN = "Construction Signs"
    MOTORIZED_SCOOTER = "Motorized Scooter"
    OTHER_VEHICLE_UNCOMMON = "Other Vehicle - Uncommon"
    OTHER_VEHICLE_CONSTRUCTION = "Other Vehicle - Construction Vehicle"
    TOWED_OBJECT = "Towed Object"
    PERSONAL_MOBILITY_DEVICE = "Personal Mobility Device"
    TRAM_SUBWAY = "Tram / Subway"
    TRAIN = "Train"
    ANIMALS_OTHER = "Animals - Other"
    SEMI_TRUCK = "Semi-truck"
    OTHER_VEHICLE_PEDICAB = "Other Vehicle - Pedicab"
    EMERGENCY_VEHICLE = "Emergency Vehicle"
    ANIMALS_BIRD = "Animals - Bird"

    @classmethod
    def get_index(cls, classtype) -> int:
        """Returns the id of a label class

        Args:
            classtype: Member of LabelClass

        Returns:
            id of classtype
        """
        return list(cls).index(classtype)


CLASS_COLORMAP = {
    LabelClass.CAR: (0, 255, 0),
    LabelClass.PICKUP: (50, 205, 50),
    LabelClass.MEDIUM_SIZED_TRUCK: (144, 238, 168),
    LabelClass.BUS: (152, 251, 152),
    LabelClass.OTHER_VEHICLE_UNCOMMON: (143, 188, 143),
    LabelClass.OTHER_VEHICLE_CONSTRUCTION: (0, 250, 154),
    LabelClass.TOWED_OBJECT: (46, 139, 87),
    LabelClass.SEMI_TRUCK: (30, 144, 255),
    LabelClass.OTHER_VEHICLE_PEDICAB: (100, 149, 237),
    LabelClass.EMERGENCY_VEHICLE: (255, 0, 0),
    LabelClass.BICYCLE: (148, 0, 211),
    LabelClass.MOTORCYCLE: (153, 50, 204),
    LabelClass.MOTORIZED_SCOOTER: (186, 85, 211),
    LabelClass.PERSONAL_MOBILITY_DEVICE: (128, 0, 128),
    LabelClass.PEDESTRIAN: (250, 128, 114),
    LabelClass.PEDESTRIAN_WITH_OBJECT: (144, 238, 144),
    LabelClass.TEMPORARY_CONSTRUCTION_BARRIER: (220, 20, 60),
    LabelClass.CONES: (255, 69, 0),
    LabelClass.SIGNS: (255, 165, 0),
    LabelClass.PYLON: (255, 160, 122),
    LabelClass.ROLLING_CONTAINER: (255, 127, 80),
    LabelClass.ROAD_BARRIER: (178, 34, 34),
    LabelClass.CONSTRUCTION_SIGN: (128, 0, 0),
    LabelClass.TRAM_SUBWAY: (192, 192, 192),
    LabelClass.TRAIN: (220, 220, 220),
    LabelClass.ANIMALS_OTHER: (255, 215, 0),
    LabelClass.ANIMALS_BIRD: (218, 165, 32),
}


@dataclass(frozen=True)
class LabelData:
    """Class for holding frame-level label data

    Args:
        uids: List of unique object identifier strings
        centroids: N x [x, y, z] centroids in vehicle frame as float32 tensor
        yaws: Yaw rotations in radians as float32 tensor
        boxes: N x [length, width, height] boxes as a float32 tensor
    """

    uids: List[str]
    centroids: torch.Tensor
    yaws: torch.Tensor
    boxes: torch.Tensor

    @property
    def rank(self):
        return self.centroids.size(0)


def pose_dict_to_transform(pose_dict: Dict) -> Transform3d:
    """Convert a pose dict to a Transform3d object

    Args:
        pose_dict: Pose dictionary like {position: {x, y, z}, heading: {w, x, y, z}}

    Returns:
        Corresponding transform3d transformation.
    """
    p, h = pose_dict["position"], pose_dict["heading"]
    position = torch.tensor([p["x"], p["y"], p["z"]], dtype=torch.float32)
    heading_ = torch.tensor([h["w"], h["x"], h["y"], h["z"]], dtype=torch.float32)
    heading = quaternion_invert(heading_)

    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_matrix(heading)

    # Return composed transformation
    return Transform3d().rotate(rot_matrix).translate(*position.T)


def label_dataframe_to_dict(
    labels: pd.DataFrame, world_to_vehicle: Transform3d
) -> Iterator[Tuple[LabelClass, LabelData]]:
    """Applies pose transform to coordinates and yaw and re-packages dataframe into a dictionary of class -> label_data.

    Args:
        labels: Base pandaset label dataframe
        world_to_vehicle: 4x4 rigid pose transform (world -> ego).

    Yields:
        Tuple of (label_class, label_data)
    """
    classes = labels.label.unique()
    pose_mat = world_to_vehicle.inverse().get_matrix()

    # The label yaws are defined relative to the ego "forward", regardless of what axis that maps onto.
    # Therefore, we need to compensate the pi/2 rotation we embedded into the transform.
    _, _, ego_yaw = matrix_to_euler_angles(pose_mat[0, :3, :3], "XYZ") + (pi / 2)
    for k in classes:
        # Filter labels by class
        class_labels = labels.query(f"`label` == '{k}'")

        # Convert relevant columns to tensors
        labels_yaw_ego = torch.from_numpy(class_labels["yaw"].to_numpy()).float()
        labels_x = torch.from_numpy(class_labels["position.x"].to_numpy()).float()
        labels_y = torch.from_numpy(class_labels["position.y"].to_numpy()).float()
        labels_z = torch.from_numpy(class_labels["position.z"].to_numpy()).float()

        # Project centroids from world to vehicle
        labels_xyz_world = torch.stack((labels_x, labels_y, labels_z), dim=-1)
        labels_xyz_veh = world_to_vehicle.transform_points(labels_xyz_world)

        # Compensate ego-yaw
        labels_yaw = labels_yaw_ego + ego_yaw

        # Dimensions
        labels_lwh = torch.tensor(
            class_labels[["dimensions.x", "dimensions.y", "dimensions.z"]].values,
            dtype=torch.float32,
        )

        # Package into dataclass
        yield (
            LabelClass(k),
            LabelData(
                uids=class_labels.uuid.to_list(),
                centroids=labels_xyz_veh,
                yaws=labels_yaw,
                boxes=labels_lwh[:, [1, 0, 2]],  # Flip axes onto LFU convention
            ),
        )
