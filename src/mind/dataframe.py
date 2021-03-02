from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from libs.pandas.cache import pd_cache


@pd_cache(cache_dir='../cache')
def load_behaviours_df(base_dir: Union[Path, str], drop_no_hist: bool = True) -> pd.DataFrame:
    base_dir = Path(base_dir)

    df_train = _load_behaviours_df(base_dir / 'train/behaviors.tsv')
    df_val = _load_behaviours_df(base_dir / 'valid/behaviors.tsv')
    if (base_dir / 'test').exists():
        df_test = _load_behaviours_df(base_dir / 'test/behaviors.tsv')
    else:
        df_test = pd.DataFrame()

    df_train['split'] = 'train'
    df_val['split'] = 'valid'
    df_test['split'] = 'test'

    df: pd.DataFrame = pd.concat((df_train, df_val, df_test), ignore_index=True)

    df['time'] = pd.to_datetime(df['time'])
    df['histories'] = df['histories'].fillna('').str.split()
    df['candidates'] = df['impressions'].str.strip() \
        .str.replace(r'-[01]', '', regex=True) \
        .str.split()
    df['labels'] = None  # Test has no labels.
    df.loc[df['split'] != 'test', 'labels'] = df[df['split'] != 'test']['impressions'].str.strip() \
        .str.replace(r'N\d*-', '', regex=True) \
        .str.split() \
        .apply(lambda x: np.array(x).astype(bool))
    df = df.drop(columns=['impressions'])

    if drop_no_hist:
        df = df[df['histories'].apply(len) > 0]

    return df


def _load_behaviours_df(tsv_path: Union[Path, str]):
    df = pd.read_table(tsv_path, header=None, names=[
        'b_id',
        'u_id',
        'time',
        'histories',
        'impressions',
    ])
    return df


@pd_cache(cache_dir='../cache')
def load_news_df(base_dir: Union[Path, str]) -> pd.DataFrame:
    base_dir = Path(base_dir)

    df_train = _load_news_df(base_dir / 'train/news.tsv')
    df_val = _load_news_df(base_dir / 'valid/news.tsv')
    if (base_dir / 'test').exists():
        df_test = _load_news_df(base_dir / 'test/news.tsv')
    else:
        df_test = pd.DataFrame()

    df: pd.DataFrame = pd.concat((df_train, df_val, df_test), ignore_index=True)
    df = df.drop_duplicates(subset=['n_id'])

    df['abstract'] = df['abstract'].fillna('')

    # df['category_label'] = LabelEncoder().fit_transform(df['category'])
    # df['subcategory_label'] = LabelEncoder().fit_transform(df['subcategory'])

    return df


def _load_news_df(tsv_path):
    df = pd.read_table(
        tsv_path,
        header=None,
        names=[
            'n_id',
            'category',
            'subcategory',
            'title',
            'abstract',
            'url',
        ],
        usecols=range(6),
    )
    return df


@pd_cache(cache_dir='../cache')
def load_popularity_df(base_dir) -> pd.DataFrame:
    df_b = load_behaviours_df(base_dir, drop_no_hist=False)
    df_b = df_b[df_b['split'] != 'test']

    df_imp_flat = pd.concat((
        df_b['candidates'].explode(),
        df_b['labels'].explode(),
    ), axis=1).reset_index(drop=True)
    df_imp_flat['labels'] = df_imp_flat['labels'].astype(np.uint8)

    df_p: pd.DataFrame = df_imp_flat.groupby('candidates').agg({'labels': [np.mean]})
    df_p.columns = ['popularity']

    return df_p
