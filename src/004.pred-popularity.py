from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, AutoTokenizer

from libs.pytorch_lightning.util import load_pretrained_dict
from mind.dataframe import load_behaviours_df, load_news_df, load_popularity_df_test
from mind.params import Params, DataParams
from mind.popularity.dataset import PopularityDataset, PopularityCollate


def load_model(ckpt_path: str):
    state_dict = load_pretrained_dict(ckpt_path)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().cuda()

    return model


def get_loader(params: DataParams):
    df_n = load_news_df(params.mind_path)
    df = load_popularity_df_test(params.mind_path)
    df = df.merge(df_n, left_index=True, right_index=True, how='left')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    loader = DataLoader(
        PopularityDataset(df),
        batch_size=params.batch_size,
        collate_fn=PopularityCollate(tokenizer, is_test=True),
        shuffle=False,
        # num_workers=cpu_count(),
        pin_memory=True,
    )

    return loader


def make_popularity_sub(logits: np.ndarray):
    df_b = load_behaviours_df(
        '../data/mind-large',
        drop_no_hist=False,
    )
    df_b = df_b[df_b['histories'].apply(len) == 0]
    df_b = df_b[df_b['split'] == 'test']

    cand_sizes = df_b['candidates'].apply(len)
    slices = np.concatenate(([0], np.cumsum(cand_sizes.values)))
    slices = [slice(a, b) for a, b in zip(slices, slices[1:])]

    assert len(df_b['b_id'].values) == len(slices), f"{len(df_b['b_id'].values)}, {len(slices)}"

    sub_rows = []
    for b_id, s in tqdm(zip(df_b['b_id'].values, slices), total=len(df_b)):
        rank = (logits[s] * -1).argsort().argsort() + 1
        rank = ','.join(rank.astype(str))
        sub_rows.append(f'{b_id} [{rank}]')

    return pd.DataFrame(
        index=df_b['b_id'],
        data=sub_rows,
        columns=['preds'],
    )


@torch.no_grad()
def main():
    ckpt_path = '/mnt/ssdnfs/vfa-ruby/akirasosa/experiments/011_popularity/1614587740/checkpoints/last.ckpt'
    params = Params.load('./params/popularity/001.yaml')

    model = load_model(ckpt_path)
    loader = get_loader(params.data_params)

    preds = []
    for batch in tqdm(loader):
        batch = move_data_to_device(batch, model.device)
        out = model.forward(**batch)
        out = out.logits.cpu().numpy().reshape(-1)
        preds.append(out)
    preds = np.concatenate(preds)

    out_dir = Path('../tmp')
    out_dir.mkdir(exist_ok=True)

    df_sub = make_popularity_sub(preds)
    df_sub.to_parquet(out_dir / 'sub_popularity.pqt')


if __name__ == '__main__':
    main()
