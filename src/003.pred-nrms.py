from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from libs.pytorch_lightning.util import load_pretrained_dict
from mind.dataframe import load_behaviours_df
from mind.main.dataset import get_test_dataset, MINDCollateVal
from mind.params import Params, DataParams, ModuleParams
from models.nrms import NRMS, ContentsEncoder


def load_model(ckpt_path: str, params: ModuleParams):
    state_dict = load_pretrained_dict(ckpt_path)

    model = NRMS(
        pretrained_model_name=params.pretrained_model_name,
        sa_pretrained_model_name=params.sa_pretrained_model_name,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().cuda()

    return model


@torch.no_grad()
def get_loader(params: DataParams, encoder: ContentsEncoder):
    tokenizer = AutoTokenizer.from_pretrained(params.pretrained_model_name)
    dateset = get_test_dataset(
        base_dir=params.mind_path,
        tokenizer=tokenizer,
    )

    encoder = encoder.eval()
    inputs = dateset.uniq_news_inputs
    feats = {
        k: encoder.forward(move_data_to_device(v, torch.device('cuda'))).squeeze().cpu()
        for k, v in tqdm(inputs.items(), desc='Encoding val candidates')
    }
    dateset.news_feature_map = feats

    loader = DataLoader(
        dateset,
        batch_size=64,
        collate_fn=MINDCollateVal(is_test=True),
        shuffle=False,
        pin_memory=True,
    )

    return loader


def make_main_sub(logits: np.ndarray):
    df_b = load_behaviours_df(
        '../data/mind-large',
        drop_no_hist=True,
    )
    df_b = df_b[df_b['split'] == 'test']

    cand_sizes = df_b['candidates'].apply(len)
    slices = np.concatenate(([0], np.cumsum(cand_sizes.values)))
    slices = [slice(a, b) for a, b in zip(slices, slices[1:])]

    assert len(df_b['b_id'].values) == slices

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
    ckpt_path = '/mnt/ssdnfs/vfa-ruby/akirasosa/experiments/010_mind_nrms/1612799645/checkpoints/epoch=1-step=328001.ckpt'
    params = Params.load('./params/main/002.yaml')

    model = load_model(ckpt_path, params.module_params)
    loader = get_loader(params.data_params, model.encoder)

    preds = []
    for batch in tqdm(loader):
        batch = move_data_to_device(batch, torch.device('cuda'))
        logits = model.forward(batch)
        logits = logits.cpu().numpy().reshape(-1)
        preds.append(logits)
    preds = np.concatenate(preds)

    out_dir = Path('../tmp')
    out_dir.mkdir(exist_ok=True)

    df_sub = make_main_sub(preds)
    df_sub.to_parquet(out_dir / 'sub_nrms.pqt')


if __name__ == '__main__':
    main()
