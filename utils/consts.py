import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import Tuple, List
from tqdm import tqdm
import os

true, false, null = True, False, None


class DataParams:

    def __init__(self,
                 dims: int = 64,
                 nheads: int = 8,
                 scale: int = 1,
                 nblocks: int = 4,
                 nl=F.silu,
                 drop_rate: float = .1,
                 eps: float = 1e-6,
                 batch_first: bool = false,
                 batch_size: int = 16,
                 seq_len: int = 32,
                 max_token: int = 5000,
                 vocab: int = 28782,
                 device: str = "cuda"
                 ):
        assert dims % nheads == 0

        self.dims = dims

        self.h = nheads
        self.sd = self.dims // self.h

        self.blcks = nblocks

        self.scale = scale
        self.nl = nl

        self.dr = drop_rate
        self.eps = eps

        self.bf = batch_first
        self.bs = batch_size
        self.seq = seq_len
        self.max_toks = max_token
        self.vocab = vocab

        self.device = device


model_args = DataParams(
    dims=200,
    drop_rate=.3,
    nheads=2,
    nblocks=2)

if __name__ == "__main__":
    pass
