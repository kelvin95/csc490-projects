import torch
from torch import nn

from prediction.model import PredictionModel, PredictionModelConfig


class ProbabilisticDecoder(nn.Module):
    ## write the code for the new model here
    pass


class ProbabilisticModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._decoder = ProbabilisticModel()
