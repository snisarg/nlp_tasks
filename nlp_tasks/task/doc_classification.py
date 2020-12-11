import hydra
import pytorch_lightning as pl

from nlp_tasks import ModelConf


class DummyClassificationTask(pl.LightningModule):
    def __init__(self, model: ModelConf):
        self.model = hydra.utils.instantiate(model)
