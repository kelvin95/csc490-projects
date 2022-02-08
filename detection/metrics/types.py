from dataclasses import dataclass

from detection.types import Detections


@dataclass
class EvaluationFrame:
    """Dataclass to store the evaluation inputs for one frame."""

    detections: Detections
    labels: Detections
