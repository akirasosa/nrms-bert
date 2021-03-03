from dataclasses import dataclass
from functools import cached_property
from logging import getLogger, FileHandler
from pathlib import Path
from time import time
from typing import Dict, Any
from typing import Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from transformers import DistilBertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from libs.pytorch_lightning.logging import configure_logging
from libs.torch.avg_meter import AverageMeter
from mind.params import ModuleParams, Params
from mind.popularity.data_module import PopularityDataModule


@dataclass(frozen=True)
class AverageMeterSet:
    train_loss: AverageMeter = AverageMeter()
    val_loss: AverageMeter = AverageMeter()


class PLModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=1,
        )
        self.am = AverageMeterSet()
        self.total_processed = 0

    def training_step(self, batch, batch_idx):
        X, y = batch
        out: SequenceClassifierOutput = self.model.forward(**X, labels=y)
        with torch.no_grad():
            n_processed = len(out.logits)
            self.am.train_loss.update(out.loss.detach(), n_processed)
            self.total_processed += n_processed
        return out.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, y = batch
        out: SequenceClassifierOutput = self.model.forward(**X, labels=y)
        n_processed = len(out.logits)
        self.am.val_loss.update(out.loss, n_processed)

    def training_epoch_end(self, outputs: Sequence[Any]):
        self.log('train_loss', self.am.train_loss.compute())

    @torch.no_grad()
    def validation_epoch_end(self, outputs: Sequence[Any]):
        self.log('val_loss', self.am.val_loss.compute())

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        sched = {
            'scheduler': OneCycleLR(
                opt,
                max_lr=self.hp.lr,
                total_steps=len(self.train_dataloader()) * self.trainer.max_epochs,
            ),
            'interval': 'step',
        }
        return [opt], [sched]
        # return [opt]

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams.from_dict(dict(self.hparams))


def train(params: Params):
    seed_everything(params.d.seed)

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name=f'011_popularity',
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(params.pretty())

    callbacks = [
        LearningRateMonitor(),
    ]
    if params.t.checkpoint_callback:
        callbacks.append(
            ModelCheckpoint(
                monitor=None,
                verbose=True,
            ),
        )
    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=params.t.precision,
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=params.t.weights_save_path,
        checkpoint_callback=params.t.weights_save_path is not None,
        callbacks=callbacks,
        deterministic=True,
        benchmark=True,
        accumulate_grad_batches=params.t.accumulate_grad_batches,
        val_check_interval=params.t.val_check_interval,
    )
    net = PLModule(params.m.to_dict())
    dm = PopularityDataModule(params.d)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    if params.do_cv:
        for p in params.copy_for_cv():
            train(p)
    else:
        train(params)
