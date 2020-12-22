from typing import Any

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from hydra_text.text.models.manual_additions import ModelConf
from hydra_text.text.models.transformer import RobertaModelConf
from omegaconf import MISSING

from nlp_tasks.model.transformers import DocModelConf


@dataclass
class TaskConf:
    data: Any = MISSING
    model: ModelConf = MISSING
    optim: Any = MISSING
    transforms: Any = MISSING

@dataclass
class TrainerConf:
    """ Can come from Lightning"""
    pass

@dataclass
class Config:
    task: TaskConf = MISSING
    trainer: TrainerConf = MISSING


cs = ConfigStore.instance()
cs.store(group="task", name="task", node=TaskConf)
cs.store(group="task/model", name="roberta_model", node=RobertaModelConf)
cs.store(group="task/model", name="doc_model", node=DocModelConf)
cs.store(group="trainer", name="trainer", node=TrainerConf)
cs.store(group="nlp_tasks", name="config", node=Config, package="_global_")
