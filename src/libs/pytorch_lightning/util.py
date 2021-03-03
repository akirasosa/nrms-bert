from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch


def load_pretrained_dict(ckpt_path: Union[str, Path]) -> OrderedDict:
    ckpt = torch.load(ckpt_path)

    if any(k.startswith('ema_model') for k in ckpt['state_dict'].keys()):
        prefix = 'ema_model'
    else:
        prefix = 'model'

    new_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if not k.startswith(f'{prefix}.'):
            continue
        new_dict[k[len(f'{prefix}.'):]] = v

    return new_dict
