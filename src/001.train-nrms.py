from dataclasses import dataclass
from functools import cached_property
from logging import getLogger, FileHandler
from pathlib import Path
from time import time
from typing import Dict, Sequence, Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.utilities import move_data_to_device
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from libs.pytorch_lightning.logging import configure_logging
from libs.torch.avg_meter import AverageMeter
from libs.torch.metrics import ndcg_score
from mind.batch import MINDBatch
from mind.data_module import MINDDataModule
from mind.dataset import MINDDatasetVal
from mind.params import ModuleParams, Params
from models.nrms import NRMS


@dataclass(frozen=True)
class AverageMeterSet:
    train_loss: AverageMeter = AverageMeter()
    val_loss: AverageMeter = AverageMeter()
    val_roc: AverageMeter = AverageMeter()
    val_ndcg10: AverageMeter = AverageMeter()


class PLModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        self.model = NRMS(
            pretrained_model_name=self.hp.pretrained_model_name,
            sa_pretrained_model_name=self.hp.sa_pretrained_model_name,
        )
        self.am = AverageMeterSet()
        self.total_processed = 0

    def training_step(self, batch: MINDBatch, batch_idx):
        loss, y_score = self.model.forward(batch)
        with torch.no_grad():
            n_processed = batch['batch_cand'].max() + 1
            self.am.train_loss.update(loss.detach(), n_processed)
            self.total_processed += n_processed
        return loss

    @torch.no_grad()
    def validation_step(self, batch: MINDBatch, batch_idx):
        loss, y_score = self.model.forward(batch)
        y_true = batch['targets']
        n_processed = batch['batch_cand'].max() + 1

        for n in range(n_processed):
            mask = batch['batch_cand'] == n
            s, t = y_score[mask], y_true[mask]
            s = torch.softmax(s, dim=0)
            self.am.val_roc.update(auroc(s, t))
            self.am.val_ndcg10.update(ndcg_score(s, t))
        self.am.val_loss.update(loss, n_processed)

    def training_epoch_end(self, outputs: Sequence[Any]):
        self.log('train_loss', self.am.train_loss.compute())

    @torch.no_grad()
    def validation_epoch_end(self, outputs: Sequence[Any]):
        self.log('val_loss', self.am.val_loss.compute())
        self.log('val_roc', self.am.val_roc.compute())
        self.log('val_ndcg10', self.am.val_ndcg10.compute())

    @torch.no_grad()
    def on_validation_epoch_start(self):
        # Pre compute feature of uniq candidates in val to save time.
        val_dataset = cast(MINDDatasetVal, self.val_dataloader().dataset)

        if self.total_processed == 0:
            val_dataset.init_dummy_feature_map(self.model.encoder.dim)
            return

        encoder = self.model.encoder.eval()
        inputs = val_dataset.uniq_news_inputs
        feats = {
            k: encoder.forward(move_data_to_device(v, self.device)).squeeze().cpu()
            for k, v in tqdm(inputs.items(), desc='Encoding val candidates')
        }
        val_dataset.news_feature_map = feats

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        # sched = {
        #     'scheduler': OneCycleLR(
        #         opt,
        #         max_lr=self.hp.lr,
        #         total_steps=len(self.train_dataloader()) * self.trainer.max_epochs,
        #     ),
        #     'interval': 'step',
        # }
        # return [opt], [sched]
        return [opt]

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams.from_dict(dict(self.hparams))


def train(params: Params):
    seed_everything(params.d.seed)

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name=f'010_mind_nrms',
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(params.pretty())

    callbacks = [
        # LearningRateMonitor(),
        # EarlyStopping(
        #     monitor='val_0_acc',
        #     patience=15,
        #     mode='max'
        # ),
    ]
    if params.t.checkpoint_callback:
        callbacks.append(
            ModelCheckpoint(
                monitor='val_roc',
                save_last=True,
                verbose=True,
                mode='max',
            )
        )

    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=params.t.precision,
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=params.t.weights_save_path,
        checkpoint_callback=params.t.checkpoint_callback,
        callbacks=callbacks,
        deterministic=True,
        benchmark=True,
        accumulate_grad_batches=params.t.accumulate_grad_batches,
        val_check_interval=params.t.val_check_interval,
    )
    net = PLModule(params.m.to_dict())
    dm = MINDDataModule(params.d)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    train(params)
