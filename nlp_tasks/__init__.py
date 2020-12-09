from typing import Any

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from hydra_text.text.decoder.mlp import ActivationFnConf, MLPDecoderConf, GeLUConf
from hydra_text.text.representation.transformers import TransformerLayerConf
from omegaconf import MISSING
from text.decoder.mlp import GeLU


@dataclass
class ModelConf:
    pass


@dataclass
class TransformerModelConf(ModelConf):
    activation_fn: Any = GeLUConf()
    transformer: TransformerLayerConf = TransformerLayerConf(attention=None, residual_mlp=None)
    mlp: MLPDecoderConf = MLPDecoderConf(in_dim=10, out_dim=2, bias=False, hidden_dims=[1], activation=GeLU)


@dataclass
class TaskConf:
    model: ModelConf = MISSING


@dataclass
class TrainerConf:
    """ Can come from Lightning"""
    pass

@dataclass
class Config:
    task: TaskConf
    trainer: TrainerConf


cs = ConfigStore.instance()
cs.store(group="task", name="task", node=TaskConf)
cs.store(group="task/model", name="transformer_model", node=TransformerModelConf)
cs.store(group="task/model/activation_fn", name="gelu", node=GeLUConf)
cs.store(group="trainer", name="trainer", node=TrainerConf)
cs.store(name="config", node=Config)
