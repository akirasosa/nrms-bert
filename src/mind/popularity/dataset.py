from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


@dataclass
class PopularityDataset(Dataset):
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
class PopularityCollate:
    tokenizer: PreTrainedTokenizerFast
    is_test: bool = False

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
        if self.is_test:
            return X

        y = torch.tensor(y).float()
        return X, y
