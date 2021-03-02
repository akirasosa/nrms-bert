from typing import TypedDict, Optional, Union

import torch
from transformers import BatchEncoding


class ContentsEncoded(TypedDict):
    title: BatchEncoding
    abstract: BatchEncoding
    category: torch.Tensor
    subcategory: torch.Tensor
    abstract_tfidf_40: torch.Tensor
    title_tfidf_40: torch.Tensor


class MINDBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Union[ContentsEncoded, torch.Tensor]
    x_cand: Union[ContentsEncoded, torch.Tensor]
    targets: Optional[torch.Tensor]
