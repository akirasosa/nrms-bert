from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Sequence, Mapping, Union, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from mind.cbf.batch import MINDBatch, ContentsEncoded
from mind.dataframe import load_behaviours_df, load_news_df


@dataclass
class MINDDatasetTrain(Dataset):
    df_behaviours: pd.DataFrame
    df_news: pd.DataFrame
    n_neg: int = 4
    hist_size: int = 50

    def __getitem__(self, idx):
        bhv = self.df_behaviours.iloc[idx]

        histories = bhv['histories']
        candidates = bhv['candidates']
        labels = bhv['labels']

        histories = np.random.permutation(histories)[:self.hist_size]
        candidates, labels = self._sample_candidates(candidates, labels)

        histories = self.df_news.loc[histories]
        candidates = self.df_news.loc[candidates]
        labels = labels.argmax()

        return histories, candidates, labels

    def __len__(self):
        return len(self.df_behaviours)

    def _sample_candidates(self, candidates, labels):
        pos_id = np.random.permutation(np.where(labels)[0])[0]

        neg_ids = np.array([]).astype(int)
        while len(neg_ids) < self.n_neg:
            neg_ids = np.concatenate((
                neg_ids,
                np.random.permutation(np.where(~labels)[0]),
            ))
        neg_ids = neg_ids[:self.n_neg]

        indices = np.concatenate(([pos_id], neg_ids))
        indices = np.random.permutation(indices)
        candidates = candidates[indices]
        labels = labels[indices]
        return candidates, labels


@dataclass
class MINDDatasetVal(Dataset):
    df_behaviours: pd.DataFrame
    df_news: pd.DataFrame
    tokenizer: InitVar[PreTrainedTokenizer]
    hist_size: int = 50
    news_feature_map: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self, tokenizer):
        self._uniq_news_inputs = self._make_uniq_news_inputs(tokenizer)

    def __getitem__(self, idx):
        assert self.news_feature_map, 'news_feature_map is empty. Set it before to get an item.'

        bhv = self.df_behaviours.iloc[idx]

        # TODO consider more
        histories = bhv['histories'][:self.hist_size]
        candidates = bhv['candidates']
        labels = bhv['labels']

        # Use precomputed features.
        histories = torch.stack([
            self.news_feature_map[idx]
            for idx in histories
        ], dim=0)
        candidates = torch.stack([
            self.news_feature_map[idx]
            for idx in candidates
        ], dim=0)

        return histories, candidates, labels

    def __len__(self):
        return len(self.df_behaviours)

    def init_dummy_feature_map(self, dim: int):
        self.news_feature_map = {
            k: torch.randn(dim)
            for k, v in self.uniq_news_inputs.items()
        }

    @property
    def uniq_news_inputs(self) -> Mapping[str, ContentsEncoded]:
        return self._uniq_news_inputs

    def _make_uniq_news_inputs(self, tokenizer) -> Mapping[str, ContentsEncoded]:
        histories = np.concatenate(self.df_behaviours['histories'].values)
        candidates = np.concatenate(self.df_behaviours['candidates'].values)

        indices = set(histories) | set(candidates)
        inputs = self.df_news.loc[indices].to_dict('records')
        inputs = [
            {
                'title': tokenizer(
                    x['title'],
                    x['category'],
                    return_tensors='pt',
                    return_token_type_ids=False,
                    truncation=True,
                ),
                # 'abstract': tokenizer(
                #     x['abstract'],
                #     return_tensors='pt',
                #     return_token_type_ids=False,
                #     truncation=True,
                # ),
                # 'category': torch.tensor([x['category_label']]),
                # 'subcategory': torch.tensor([x['subcategory_label']]),
            }
            for x in inputs
        ]

        return dict(zip(indices, inputs))


def _make_batch_assignees(items: Sequence[Sequence[Any]]) -> torch.Tensor:
    sizes = torch.tensor([len(x) for x in items])
    batch = torch.repeat_interleave(torch.arange(len(items)), sizes)
    return batch


@dataclass
class MINDCollateTrain:
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        histories, candidates, targets = zip(*batch)

        batch_hist = _make_batch_assignees(histories)
        batch_cand = _make_batch_assignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))

        return MINDBatch(
            batch_hist=batch_hist,
            batch_cand=batch_cand,
            x_hist=x_hist,
            x_cand=x_cand,
            targets=torch.tensor(targets),
        )

    def _tokenize(self, x: List[str]):
        return self.tokenizer(
            x,
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True,
            truncation=True,
        )

    def _tokenize_df(self, df: pd.DataFrame):
        return {
            'title': self._tokenize(df[['title', 'category']].values.tolist()),
            # 'title': self._tokenize(df['title'].values.tolist()),
            # 'abstract': self._tokenize(df['abstract'].values.tolist()),
            # 'category': torch.from_numpy(df['category_label'].values).long(),
            # 'subcategory': torch.from_numpy(df['subcategory_label'].values).long(),
        }


@dataclass
class MINDCollateVal:
    is_test: bool = False

    def __call__(self, batch):
        # It gets precomputed inputs.
        histories, candidates, targets = zip(*batch)

        batch_hist = _make_batch_assignees(histories)
        batch_cand = _make_batch_assignees(candidates)

        x_hist = torch.cat(histories)
        x_cand = torch.cat(candidates)
        if self.is_test:
            targets = None
        else:
            targets = np.concatenate(targets)
            targets = torch.from_numpy(targets)

        return MINDBatch(
            batch_hist=batch_hist,
            batch_cand=batch_cand,
            x_hist=x_hist,
            x_cand=x_cand,
            targets=targets,
        )


def _load_df(base_dir: Union[str, Path], split: str):
    df_b = load_behaviours_df(base_dir)
    df_b = df_b[df_b['split'] == split]
    df_b = df_b[['histories', 'candidates', 'labels']]

    df_n = load_news_df(base_dir)
    df_n['category'] = df_n['category'] + ' > ' + df_n['subcategory']
    df_n = df_n[[
        'title',
        'category',
        # 'abstract',
        # 'category_label',
        # 'subcategory_label',
    ]]
    return df_b, df_n


def get_train_dataset(base_dir: Union[str, Path]) -> MINDDatasetTrain:
    df_b, df_n = _load_df(base_dir, 'train')

    return MINDDatasetTrain(
        df_behaviours=df_b,
        df_news=df_n,
    )


def get_val_dataset(base_dir: Union[str, Path], tokenizer: PreTrainedTokenizer) -> MINDDatasetVal:
    df_b, df_n = _load_df(base_dir, 'valid')

    return MINDDatasetVal(
        df_behaviours=df_b,
        df_news=df_n,
        tokenizer=tokenizer,
    )


def get_test_dataset(base_dir: Union[str, Path], tokenizer: PreTrainedTokenizer) -> MINDDatasetVal:
    df_b, df_n = _load_df(base_dir, 'test')

    return MINDDatasetVal(
        df_behaviours=df_b,
        df_news=df_n,
        tokenizer=tokenizer,
    )


# For dev
def get_train_sample_batch():
    ds = get_train_dataset(base_dir='../data/mind-demo')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    collate_fn = MINDCollateTrain(tokenizer)
    return collate_fn([ds[0], ds[1]])


# %%
if __name__ == '__main__':
    # %%
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # %%
    ds = get_train_dataset(base_dir='../data/mind-demo')

    # %%
    collate_fn = MINDCollateTrain(tokenizer)
    batch = collate_fn([ds[0], ds[1]])
    # %%
    print(type(batch['x_hist']) is dict)
    print(type(batch['x_cand']) is dict)

    # %%
    print(batch['x_hist']['category'].shape)
    print(batch['x_hist']['abstract_tfidf_40'].shape)

    # %%
    ds_val = get_val_dataset(base_dir='../data/mind-demo', tokenizer=tokenizer)
    ds_val.init_dummy_feature_map(100)
    collate_fn = MINDCollateVal()
    batch = collate_fn([ds_val[0], ds_val[1]])

    # %%
    print(type(batch['x_hist']) is torch.Tensor)
    print(type(batch['x_cand']) is torch.Tensor)
    ds_val.uniq_news_inputs

    # %%
    ds_test = get_test_dataset(base_dir='../data/mind-large', tokenizer=tokenizer)
    ds_test.init_dummy_feature_map(100)
    collate_fn = MINDCollateVal()
    batch = collate_fn([ds_test[0], ds_test[1]])
