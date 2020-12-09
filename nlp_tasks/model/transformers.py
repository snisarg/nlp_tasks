from typing import Any

import torch.nn as nn
import hydra

from hydra_text.text.decoder.mlp import ActivationFnConf, MLPDecoderConf
from hydra_text.text.representation.transformers import TransformerLayerConf


class TransformerModel(nn.Module):
    def __init__(self, activation_fn: ActivationFnConf, transformer: TransformerLayerConf, mlp: MLPDecoderConf):
        self.activation_fn = hydra.utils.instantiate(activation_fn)
        self.transformer = hydra.utils.instantiate(transformer)
        self.mlp = hydra.utils.instantiate(mlp)

    def forward(self):
        pass


