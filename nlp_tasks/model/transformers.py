from dataclasses import field, dataclass
from typing import Any, Optional, List

import torch.nn as nn
import hydra

from hydra_text.text.decoder.mlp import ActivationFnConf, MLPDecoderConf
from hydra_text.text.models.manual_additions import ModelConf
from hydra_text.text.representation.transformers import TransformerLayerConf
from omegaconf import MISSING


@dataclass
class DocModelConf(ModelConf):
    _target_: str = "pytext.contrib.pytext_lib.models.DocModel"
    # word embedding config
    pretrained_embeddings_path: str = MISSING
    embedding_dim: int = MISSING
    mlp_layer_dims: Optional[List[int]] = field(default_factory=list)
    lowercase_tokens: bool = False
    skip_header: bool = True
    delimiter: str = " "
    # DocNN config
    kernel_num: int = 100
    kernel_sizes: Optional[List[int]] = field(default_factory=list)
    pooling_type: str = "max"
    dropout: float = 0.4
    # decoder config
    dense_dim: int = 0
    decoder_hidden_dims: Optional[List[int]] = field(default_factory=list)
    out_dim: int = 2


class DocModel(nn.Module):
    def __init__(
        self,
        pretrained_embeddings_path: str,
        embedding_dim: int,
        mlp_layer_dims: Optional[List[int]] = None,
        lowercase_tokens: bool = False,
        skip_header: bool = True,
        delimiter: str = " ",
        # DocNN config
        kernel_num: int = 100,
        kernel_sizes: Optional[List[int]] = [],
        pooling_type: str = "max",
        dropout: float = 0.4,
        # decoder config
        dense_dim: int = 0,
        decoder_hidden_dims: Optional[List[int]] = [],
        out_dim: int = 2,
    ):
        pass




