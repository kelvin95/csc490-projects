from turtle import forward

import torch
from torch import nn

from prediction.model import PredictionModel, PredictionModelConfig


class InteractiveDecoder(nn.Module):
    ## write the code for the new model here
    pass


class InteractiveModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._decoder = InteractiveModel()

    def forward(self, x):
        # currently the data is preprocessed to be in the actor frame. This
        # means that the relative position of each agent is lost. You may have
        # to change how to the preprocessing works to account for this
        pass
