'''
**************************************************************************
The models here are designed to work with datasets having maximum arity 6.
**************************************************************************
'''


import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math
import itertools
import torch.nn as nn

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

class RealE(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(RealE, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.w = kwargs["window_size"]
        self.nw = self.emb_dim // self.w
        self.non_linearity = kwargs["non_linearity"]
        self.ent_non_linearity = kwargs["ent_non_linearity"]
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim * self.max_arity, padding_idx=0)
        self.R_bias = torch.nn.Embedding(dataset.num_rel(), self.nw, padding_idx=0)

        self.input_dropout = torch.nn.Dropout(kwargs["input_drop"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_drop"])

    def init(self):
        self.E.weight.data[0] = torch.zeros(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.R_bias.weight.data)

    def apply_non_linearity(self, non_linearity, e):
        if non_linearity == "relu":
            return torch.relu(e)
        elif non_linearity == "tanh":
            return torch.tanh(e)
        elif non_linearity == "sigmoid":
            return torch.sigmoid(e)
        elif non_linearity == "exp":
            return torch.exp(e)
        elif non_linearity == "lrelu":
            return torch.nn.functional.leaky_relu(e)
        elif non_linearity == "srelu":
            return torch.clamp(e, 0, 1)
        elif non_linearity == "none":
            return e

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):

        r = self.R(r_idx).reshape(-1, self.max_arity, self.emb_dim)
        r_bias = self.R_bias(r_idx)

        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        entity_embs = self.input_dropout(torch.stack((e1, e2, e3, e4, e5, e6), dim=1))
        entity_embs = self.apply_non_linearity(self.ent_non_linearity, entity_embs)
        r = r.reshape(r.shape[0], r.shape[1], self.w, self.nw)
        entity_embs = entity_embs.reshape(entity_embs.shape[0], entity_embs.shape[1], self.w, self.nw)
        results = torch.sum(self.hidden_dropout(self.apply_non_linearity(self.non_linearity, torch.sum(r * entity_embs , (2, 1)) + r_bias)), dim=1)

        return results/self.nw

