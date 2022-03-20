import torch
from torch import nn

from prediction.model import PredictionModel, PredictionModelConfig


class MultiModalDecoder(nn.Module):
    ## write the code for the new model here
    pass


class MultiModalModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._decoder = MultiModalModel()
