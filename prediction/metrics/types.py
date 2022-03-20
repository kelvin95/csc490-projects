from dataclasses import dataclass

from prediction.types import Trajectories


@dataclass
class EvaluationFrame:
    """Dataclass to store the evaluation input for one frame"""

    trajectories: Trajectories
    labels: Trajectories
