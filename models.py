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

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)


class MDistMult(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MDistMult, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class MCP(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MCP, self).__init__()
        self.emb_dim = emb_dim
        self.E1 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E2 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E3 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E4 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E5 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E6 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E1.weight.data[0] = torch.ones(self.emb_dim)
        self.E2.weight.data[0] = torch.ones(self.emb_dim)
        self.E3.weight.data[0] = torch.ones(self.emb_dim)
        self.E4.weight.data[0] = torch.ones(self.emb_dim)
        self.E5.weight.data[0] = torch.ones(self.emb_dim)
        self.E6.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E1.weight.data[1:])
        xavier_normal_(self.E2.weight.data[1:])
        xavier_normal_(self.E3.weight.data[1:])
        xavier_normal_(self.E4.weight.data[1:])
        xavier_normal_(self.E5.weight.data[1:])
        xavier_normal_(self.E6.weight.data[1:])
        xavier_normal_(self.R.weight.data)

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E1(e1_idx)
        e2 = self.E2(e2_idx)
        e3 = self.E3(e3_idx)
        e4 = self.E4(e4_idx)
        e5 = self.E5(e5_idx)
        e6 = self.E6(e6_idx)
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class HSimplE(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(HSimplE, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)


    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.shift(self.E(e2_idx), int(1 * self.emb_dim/self.max_arity))
        e3 = self.shift(self.E(e3_idx), int(2 * self.emb_dim/self.max_arity))
        e4 = self.shift(self.E(e4_idx), int(3 * self.emb_dim/self.max_arity))
        e5 = self.shift(self.E(e5_idx), int(4 * self.emb_dim/self.max_arity))
        e6 = self.shift(self.E(e6_idx), int(5 * self.emb_dim/self.max_arity))
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class HypE(BaseClass):
    def __init__(self, d, emb_dim, **kwargs):
        super(HypE, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]
        self.stride = kwargs["stride"]
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.emb_dim = emb_dim
        self.max_arity = 6
        rel_emb_dim = emb_dim
        self.E = torch.nn.Embedding(d.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(d.num_rel(), rel_emb_dim, padding_idx=0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, emb_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(rel_emb_dim + self.max_arity + 1, fc1_length)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve(self, r_idx, e_idx, pos):

        e = self.E(e_idx).view(-1, 1, 1, self.E.weight.size(1))
        r = self.R(r_idx)
        x = e
        x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):
        r = self.R(r_idx)
        e1 = self.convolve(r_idx, e1_idx, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
        e2 = self.convolve(r_idx, e2_idx, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
        e3 = self.convolve(r_idx, e3_idx, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
        e4 = self.convolve(r_idx, e4_idx, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
        e5 = self.convolve(r_idx, e5_idx, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
        e6 = self.convolve(r_idx, e6_idx, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

        x = e1 * e2 * e3 * e4 * e5 * e6 * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class MTransH(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MTransH, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R1 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.R2 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.b0 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b1 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b2 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b3 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b4 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b5 = torch.nn.Embedding(dataset.num_rel(), 1)

        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.zeros(self.emb_dim)
        self.R1.weight.data[0] = torch.zeros(self.emb_dim)
        self.R2.weight.data[0] = torch.zeros(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R1.weight.data[1:])
        xavier_normal_(self.R2.weight.data[1:])
        normalize_entity_emb = F.normalize(self.E.weight.data[1:], p=2, dim=1)
        normalize_relation_emb = F.normalize(self.R1.weight.data[1:], p=2, dim=1)
        normalize_norm_emb = F.normalize(self.R2.weight.data[1:], p=2, dim=1)
        self.E.weight.data[1:] = normalize_entity_emb
        self.R1.weight.data[1:] = normalize_relation_emb
        self.R2.weight.data[1:] = normalize_norm_emb
        xavier_normal_(self.b0.weight.data)
        xavier_normal_(self.b1.weight.data)
        xavier_normal_(self.b2.weight.data)
        xavier_normal_(self.b3.weight.data)
        xavier_normal_(self.b4.weight.data)
        xavier_normal_(self.b5.weight.data)

    def pnr(self, e_idx, r_idx):
        original = self.E(e_idx)
        norm = self.R2(r_idx)
        return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms):
        r = self.R1(r_idx)
        e1 = self.pnr(e1_idx, r_idx) * self.b0(r_idx)
        e1 = e1 * ms[:,0].unsqueeze(-1).expand_as(e1)
        e2 = self.pnr(e2_idx, r_idx) * self.b1(r_idx)
        e2 = e2 * ms[:,1].unsqueeze(-1).expand_as(e2)
        e3 = self.pnr(e3_idx, r_idx) * self.b2(r_idx)
        e3 = e3 * ms[:,2].unsqueeze(-1).expand_as(e3)
        e4 = self.pnr(e4_idx, r_idx) * self.b3(r_idx)
        e4 = e4 * ms[:,3].unsqueeze(-1).expand_as(e4)
        e5 = self.pnr(e5_idx, r_idx) * self.b4(r_idx)
        e5 = e5 * ms[:,4].unsqueeze(-1).expand_as(e5)
        e6 = self.pnr(e6_idx, r_idx) * self.b5(r_idx)
        e6 = e6 * ms[:,5].unsqueeze(-1).expand_as(e6)
        x = r + e1 + e2 + e3 + e4 + e5 + e6
        x = self.hidden_drop(x)
        x = -1 * torch.norm(x, p=2, dim=1)
        return x

class RealEv1(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(RealEv1, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        self.w = kwargs["filt_w"]
        self.b = self.emb_dim // self.w
        self.non_linearity = kwargs["non_linearity"]
        self.ent_non_linearity = kwargs["ent_non_linearity"]
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)

        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim * self.max_arity, padding_idx=0)
        self.R_bias = torch.nn.Embedding(dataset.num_rel(), self.b, padding_idx=0)

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
            return self.special_relu(e)
        elif non_linearity == "none":
            return e

    def special_relu(self, e):
        e[e > 1.0] = 1.0
        e[e < 0.0] = 0.0
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

        entity_embs = torch.stack((e1, e2, e3, e4, e5, e6), dim=1)
        entity_embs = self.apply_non_linearity(self.ent_non_linearity, entity_embs)
        r = r.reshape(r.shape[0], r.shape[1], self.w, self.b)
        entity_embs = entity_embs.reshape(entity_embs.shape[0], entity_embs.shape[1], self.w, self.b)
        results = torch.sum(self.apply_non_linearity(self.non_linearity, torch.sum(r * entity_embs , (2, 1)) + r_bias), dim=1)

        return results

class RealEv3(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(RealEv3, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        self.w = kwargs["filt_w"]
        self.b = self.emb_dim // self.w
        self.non_linearity = kwargs["non_linearity"]
        self.smart_initialization = kwargs["smart_initialization"]
        self.ent_non_linearity = kwargs["ent_non_linearity"]
        self.reg = kwargs["reg"]
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.dataset = dataset
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim * self.max_arity, padding_idx=0)
        self.R_bias = torch.nn.Embedding(dataset.num_rel(), self.b, padding_idx=0)

        self.R_weights_0 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)
        self.R_weights_1 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)
        self.R_weights_2 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)
        self.R_weights_3 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)
        self.R_weights_4 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)
        self.R_weights_5 = torch.nn.Embedding(dataset.num_rel(), 1, padding_idx=0)


    def init(self):
        self.E.weight.data[0] = torch.zeros(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.R_bias.weight.data)
        if self.smart_initialization:
            self.R_weights_0.weight.data = torch.zeros(self.dataset.num_rel(), 1).cuda()
            self.R_weights_1.weight.data = torch.zeros(self.dataset.num_rel(), 1).cuda()
            self.R_weights_2.weight.data = torch.zeros(self.dataset.num_rel(), 1).cuda()
            self.R_weights_3.weight.data = torch.zeros(self.dataset.num_rel(), 1).cuda()
            self.R_weights_4.weight.data = torch.zeros(self.dataset.num_rel(), 1).cuda()
            self.R_weights_5.weight.data = torch.ones(self.dataset.num_rel(), 1).cuda()

        else:
            xavier_normal_(self.R_weights_0.weight.data)
            xavier_normal_(self.R_weights_1.weight.data)
            xavier_normal_(self.R_weights_2.weight.data)
            xavier_normal_(self.R_weights_3.weight.data)
            xavier_normal_(self.R_weights_4.weight.data)
            xavier_normal_(self.R_weights_5.weight.data)

    def loss(self):
        return self.reg * torch.sum(torch.abs(self.R_weights_0.weight) + torch.abs(self.R_weights_1.weight) + torch.abs(self.R_weights_2.weight) + torch.abs(self.R_weights_3.weight) + torch.abs(self.R_weights_4.weight) + torch.abs(self.R_weights_5.weight))

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
            return self.special_relu(e)
        elif non_linearity == "none":
            return e

    def special_relu(self, e):
        e[e > 1.0] = 1.0
        e[e < 0.0] = 0.0
        return e

    def forward_(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):

        r = self.R(r_idx).reshape(-1, self.max_arity, self.emb_dim)
        r_bias = self.R_bias(r_idx)

        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        entity_embs = torch.stack((e1, e2, e3, e4, e5, e6), dim=1)
        entity_embs = self.apply_non_linearity(self.ent_non_linearity, entity_embs)
        r = r.reshape(r.shape[0], r.shape[1], self.w, self.b)
        entity_embs = entity_embs.reshape(entity_embs.shape[0], entity_embs.shape[1], self.w, self.b)
        results = torch.sum(self.apply_non_linearity(self.non_linearity, torch.sum(r * entity_embs , (2, 1)) + r_bias), dim=1)
        return results

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        zeros = torch.zeros(e1_idx.shape).long().cuda()
        zeros.requires_grad = False

        r = torch.cat((r_idx, r_idx, r_idx, r_idx, r_idx, r_idx, r_idx, r_idx, r_idx, r_idx, r_idx), 0)
        e1 = torch.cat((e1_idx, zeros, e1_idx, zeros, e1_idx, zeros,  e1_idx, zeros, e1_idx, zeros, e1_idx), 0)
        e2 = torch.cat((zeros, e2_idx, e2_idx, zeros, e2_idx, zeros, e2_idx, zeros, e2_idx, zeros, e2_idx), 0)
        e3 = torch.cat((zeros, e3_idx, zeros, e3_idx, e3_idx, zeros, e3_idx, zeros, e3_idx, zeros, e3_idx), 0)
        e4 = torch.cat((zeros, e4_idx, zeros, e4_idx, zeros, e4_idx, e4_idx, zeros, e4_idx, zeros, e4_idx), 0)
        e5 = torch.cat((zeros, e5_idx, zeros, e5_idx, zeros, e5_idx, zeros, e5_idx, e5_idx, zeros, e5_idx), 0)
        e6 = torch.cat((zeros, e6_idx, zeros, e6_idx, zeros, e6_idx, zeros, e6_idx, zeros, e6_idx, e6_idx), 0)

        scores = self.forward_(r, e1, e2, e3, e4, e5, e6).reshape(11, -1).transpose(0, 1)

        output = self.R_weights_0(r_idx).squeeze() * scores[:, 0] * scores[:, 1] + self.R_weights_1(r_idx).squeeze() * scores[:, 2] * scores[:, 3] \
            + self.R_weights_2(r_idx).squeeze() * scores[:, 4] * scores[:, 5] + self.R_weights_3(r_idx).squeeze() * scores[:, 6] * scores[:, 7] \
            + self.R_weights_4(r_idx).squeeze() * scores[:, 8] * scores[:, 9] + self.R_weights_5(r_idx).squeeze() * scores[:, 10]

        return output


class GETD(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(GETD, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        # k has to be the max_arity + 1
        self.k = self.max_arity + 1
        self.ranks = kwargs["filt_w"]
        self.ni = emb_dim

        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.Zlist = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (self.ranks, self.ni, self.ranks)), dtype=torch.float, requires_grad=True).to(self.device)) for _ in range(self.k)])

        self.bne = torch.nn.BatchNorm1d(self.emb_dim)
        self.bnr = torch.nn.BatchNorm1d(self.emb_dim)
        self.bnw = torch.nn.BatchNorm1d(self.emb_dim)

        self.input_dropout = torch.nn.Dropout(0.46694419227220374)
        self.hidden_dropout = torch.nn.Dropout(0.18148844341064124)

    def init(self):
        self.E.weight.data[0] = torch.zeros(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)


    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, W=None):
        de = self.E.weight.shape[1]
        dr = self.R.weight.shape[1]

        if W is None:

            k = len(self.Zlist)
            Zlist = [Z for Z in self.Zlist]
            if k == 4:
                W0 = torch.einsum('aib,bjc,ckd,dla->ijkl', Zlist)
            elif k == 5:
                W0 = torch.einsum('aib,bjc,ckd,dle,ema->ijklm', Zlist)
            elif k == 6:
                W0 = torch.einsum('aib,bjc,ckd,dle,emf, fna->ijklmn', Zlist)
            elif k == 7:
                W0 = torch.einsum('aib,bjc,ckd,dle,emf,fng,goa->ijklmno', Zlist)

            W = W0.view(dr, de, de, de, de, de, de)

        r = self.bnr(self.R(r_idx))
        W_mat = torch.mm(r, W.view(r.size(1), -1))

        
        W_mat = W_mat.view(-1, de, de, de, de, de, de)
        e1, e2, e3, e4, e5, e6 = self.E(e1_idx), self.E(e2_idx), self.E(e3_idx), self.E(e4_idx), self.E(e5_idx), self.E(e6_idx),
        e1, e2, e3, e4, e5, e6 = self.bne(e1), self.bne(e2), self.bne(e3), self.bne(e4), self.bne(e5), self.bne(e6)
        e1, e2, e3, e4, e5, e6 = self.input_dropout(e1), self.input_dropout(e2), self.input_dropout(e3), self.input_dropout(e4), self.input_dropout(e5), self.input_dropout(e6)

        W_mat1 = torch.einsum('ijklmno,ij,ik,il,im,in,io->i', W_mat, e1, e2, e3, e4, e5, e6)
        pred = self.hidden_dropout(W_mat1)
        return pred

class ERMLP(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(ERMLP, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop1 = torch.nn.Dropout(self.hidden_drop_rate)
        self.hidden_drop2 = torch.nn.Dropout(self.hidden_drop_rate)
        self.hidden_drop3 = torch.nn.Dropout(self.hidden_drop_rate)
        self.hidden_drop4 = torch.nn.Dropout(self.hidden_drop_rate)

        self.non_linearity = kwargs["non_linearity"]
        # self.hidden_size = kwargs["filt_w"]
        self.fc1 = torch.nn.Linear(emb_dim * 7, 700)
        self.fc2 = torch.nn.Linear(700, 300)
        self.fc3 = torch.nn.Linear(300, 100)
        self.fc4 = torch.nn.Linear(100, 1)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

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
            return self.special_relu(e)
        elif non_linearity == "none":
            return e

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        input_layer = torch.cat((r, e1, e2, e3, e4, e5, e6), 1)
        output = self.fc1(input_layer)
        output = self.hidden_drop1(output)
        output = self.apply_non_linearity(self.non_linearity, output)
        output = self.fc2(output)
        output = self.hidden_drop2(output)
        output = self.apply_non_linearity(self.non_linearity, output)
        output = self.fc3(output)
        output = self.hidden_drop3(output)
        output = self.apply_non_linearity(self.non_linearity, output)
        output = self.fc4(output)
        output = self.hidden_drop4(output)
        output = self.apply_non_linearity(self.non_linearity, output)
        output = output.reshape(r_idx.shape[0])

        return output



