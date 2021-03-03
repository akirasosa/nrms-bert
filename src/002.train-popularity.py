from dataclasses import dataclass
from functools import cached_property
from logging import getLogger, FileHandler
from multiprocessing import cpu_count
from pathlib import Path
from time import time
from typing import Dict, Any
from typing import Optional, Union
from typing import Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

from libs.pytorch_lightning.logging import configure_logging
from libs.torch.avg_meter import AverageMeter
from mind.dataframe import load_popularity_df, load_news_df
from mind.params import DataParams
from mind.params import ModuleParams, Params


@dataclass
class MyDataset(Dataset):
    df: pd.DataFrame

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = row['title']
        category = row['category']
        subcategory = row['subcategory']
        label = row['popularity']
        return title, f'{category} > {subcategory}', label

    def __len__(self):
        return len(self.df)


@dataclass
class MyCollate:
    tokenizer: PreTrainedTokenizerFast

    def __call__(self, batch):
        x0, x1, y = zip(*batch)
        X = self.tokenizer(
            list(x0),
            list(x1),
            return_tensors='pt',
            return_token_type_ids=False,  # distilbert
            truncation=True,
            padding=True,
        )
        y = torch.tensor(y).float()
        return X, y


# noinspection PyAbstractClass
class PopularityDataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[MyDataset] = None
        self.val_dataset: Optional[MyDataset] = None
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def setup(self, stage: Optional[str] = None):
        df_p = load_popularity_df(self.params.mind_path)
        df_p = df_p[df_p['popularity'] > 0]
        df_n = load_news_df(self.params.mind_path)
        df = df_p.merge(df_n, left_index=True, right_index=True, how='left')

        if self.params.train_all:
            df_train = df
            df_val = df.iloc[:10]
        else:
            kf = KFold(
                n_splits=self.params.n_splits,
                random_state=self.params.seed,
                shuffle=True,
            )
            train_idx, val_idx = list(kf.split(df))[self.params.fold]
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

        self.train_dataset = MyDataset(df_train)
        self.val_dataset = MyDataset(df_val)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MyCollate(self.tokenizer),
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MyCollate(self.tokenizer),
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )


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
                save_last=True,
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
