from multiprocessing import cpu_count
from typing import Optional, Union, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mind.cbf.dataset import MINDDatasetTrain, MINDDatasetVal, get_train_dataset, get_val_dataset, MINDCollateTrain, \
    MINDCollateVal
from mind.params import DataParams


# noinspection PyAbstractClass
class MINDDataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[MINDDatasetTrain] = None
        self.val_dataset: Optional[MINDDatasetVal] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            params.pretrained_model_name,
            use_fast=True,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = get_train_dataset(base_dir=self.params.mind_path)
        self.val_dataset = get_val_dataset(
            base_dir=self.params.mind_path,
            # This tokenizer must be different instance from others.
            tokenizer=AutoTokenizer.from_pretrained(
                self.params.pretrained_model_name,
                use_fast=False,
            ),
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MINDCollateTrain(self.tokenizer),
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MINDCollateVal(),
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )
