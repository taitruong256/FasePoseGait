"""Pure-PyTorch implementation of the ProtoGCN backbone.

This module mirrors ``docs/ProtoGCN/protogcn/models/gcns`` but deliberately
does not depend on MMCV.  Its input is ``[N, M, T, V, C]`` and it returns the
per-person feature map plus the reconstructed dynamic graph.
"""
import copy
import numpy as np
import torch
import torch.nn as nn


EPS = 1e-4


class ProtoGraph:
    """The graph generator used by the original ProtoGCN repository."""

    def __init__(self, layout='coco', mode='random', num_filter=8,
                 init_std=.02, init_off=.04):
        if layout != 'coco':
            raise ValueError('ProtoGCN currently supports the COCO-17 layout only.')
        if mode != 'random':
            raise ValueError('This port currently supports ProtoGCN random graphs only.')
        self.num_node = 17
        self.A = np.random.randn(num_filter, self.num_node, self.num_node) * init_std + init_off


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, dropout=0., norm=True):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    """Multi-scale temporal convolution from the original ProtoGCN."""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 num_joints=25, dropout=0.,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'),
                 stride=1):
        super().__init__()
        num_branches = len(ms_cfg)
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.add_coeff = nn.Parameter(torch.zeros(num_joints))
        act = nn.ReLU()
        branches = []
        for index, cfg in enumerate(ms_cfg):
            branch_channels = rem_mid_channels if index == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_channels, 1, stride=(stride, 1)))
            elif cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), act,
                    nn.MaxPool2d((cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), act,
                    UnitTCN(branch_channels, branch_channels, cfg[0], stride, cfg[1], norm=False)))
        self.branches = nn.ModuleList(branches)
        total_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(nn.BatchNorm2d(total_channels), nn.ReLU(),
                                       nn.Conv2d(total_channels, out_channels, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        _, _, _, vertices = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], dim=-1)
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        local, global_ = out[..., :vertices], out[..., vertices]
        global_ = torch.einsum('nct,v->nctv', global_, self.add_coeff[:vertices])
        return self.drop(self.bn(self.transform(local + global_)))


class UnitGCN(nn.Module):
    """View-aware graph convolution and motion topology enhancement."""
    def __init__(self, in_channels, out_channels, adjacency, view_num=11,
                 ratio=.125, intra_act='softmax', inter_act='tanh'):
        super().__init__()
        self.num_subsets = adjacency.size(0)
        self.mid_channels = int(ratio * out_channels)
        if self.mid_channels <= 0:
            raise ValueError('ProtoGCN ratio produces zero intermediate channels.')
        self.intra_act, self.inter_act = intra_act, inter_act
        self.A = nn.Parameter(adjacency.clone())
        channels = self.mid_channels * self.num_subsets
        self.pre = nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU())
        self.post = nn.Conv2d(channels, out_channels, 1)
        self.view_conv = nn.Conv2d(in_channels, channels, 1)
        self.view_fc = nn.Linear(channels, view_num)
        self.view_mats = nn.Parameter(adjacency.clone().unsqueeze(0).repeat(view_num, 1, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, channels, 1)
        self.conv2 = nn.Conv2d(in_channels, channels, 1)
        self.down = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
                     if in_channels != out_channels else nn.Identity())
        self.bn, self.act = nn.BatchNorm2d(out_channels), nn.ReLU()

    def forward(self, x):
        n, _, t, v = x.shape
        residual = self.down(x)
        view_logits = self.view_fc(self.view_conv(x).mean(dim=(2, 3)))
        view_prob = torch.softmax(view_logits, dim=-1)
        self.last_view_logits = view_logits
        adjacency = (self.A[None, :, None, None] +
                     torch.einsum('nv,vkxy->nkxy', view_prob, self.view_mats)[:, :, None, None]) / 2
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, t, v).mean(dim=-2, keepdim=True)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, t, v).mean(dim=-2, keepdim=True)
        inter = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2)) * self.alpha[0]
        intra = torch.softmax(torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None], dim=-2) * self.beta[0]
        adjacency = (adjacency + inter + intra).squeeze(3)
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, adjacency).reshape(n, -1, t, v)
        x = self.post(x)
        graph = (inter + intra).squeeze(3).reshape(n, -1, v, v)
        return self.act(self.bn(x) + residual), graph


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, residual=True, **kwargs):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency,
                           view_num=kwargs.get('view_num', 11), ratio=kwargs.get('gcn_ratio', .125),
                           intra_act=kwargs.get('intra_act', 'softmax'), inter_act=kwargs.get('inter_act', 'tanh'))
        self.tcn = MSTCN(out_channels, out_channels, stride=stride,
                         num_joints=kwargs.get('num_joints', 25), dropout=kwargs.get('tcn_dropout', 0.),
                         ms_cfg=kwargs.get('tcn_ms_cfg', ((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1')))
        self.residual = (lambda x: 0 if not residual else x) if not residual or (in_channels == out_channels and stride == 1) else UnitTCN(in_channels, out_channels, 1, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        x, graph = self.gcn(x)
        return self.relu(self.tcn(x) + residual), graph


class PrototypeReconstructionNetwork(nn.Module):
    def __init__(self, dim, n_prototype=100, dropout=.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias=False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.memory_matrix(torch.softmax(self.query_matrix(x), dim=-1)))


class ProtoGCNBackbone(nn.Module):
    def __init__(self, in_channels=10, base_channels=96, ch_ratio=2, num_stages=10,
                 inflate_stages=(5, 8), down_stages=(5, 8), num_person=1,
                 data_bn_type='VC', graph_cfg=None, num_prototype=300, **kwargs):
        super().__init__()
        graph_cfg = graph_cfg or {'layout': 'coco', 'mode': 'random', 'num_filter': 8}
        graph = ProtoGraph(**graph_cfg)
        adjacency = torch.tensor(graph.A, dtype=torch.float32)
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * adjacency.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * adjacency.size(1))
        else:
            self.data_bn = nn.Identity()
        self.data_bn_type, self.num_person = data_bn_type, num_person
        stage_kwargs = [copy.deepcopy(kwargs) for _ in range(num_stages)]
        for key, value in kwargs.items():
            if isinstance(value, tuple) and len(value) == num_stages:
                for i in range(num_stages):
                    stage_kwargs[i][key] = value[i]
        stage_kwargs[0].pop('tcn_dropout', None)
        modules = []
        if in_channels != base_channels:
            modules.append(GCNBlock(in_channels, base_channels, adjacency.clone(), residual=False, **stage_kwargs[0]))
        current_channels, inflated = base_channels, 0
        for stage in range(2, num_stages + 1):
            if stage in inflate_stages:
                inflated += 1
            out_channels = int(base_channels * ch_ratio ** inflated + EPS)
            modules.append(GCNBlock(current_channels, out_channels, adjacency.clone(),
                                    stride=1 + (stage in down_stages), **stage_kwargs[stage - 1]))
            current_channels = out_channels
        self.gcn = nn.ModuleList(modules)
        self.out_channels = current_channels
        self.post = nn.Conv2d(current_channels, current_channels, 1)
        self.bn, self.relu = nn.BatchNorm2d(current_channels), nn.ReLU()
        self.prn = PrototypeReconstructionNetwork(current_channels, num_prototype)

    def forward(self, x):
        n, m, t, v, c = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(n, m * v * c, t))
        else:
            x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        graphs = []
        view_logits_list = []
        for block in self.gcn:
            x, graph = block(x)
            graphs.append(graph)
            view_logits_list.append(block.gcn.last_view_logits)
        x = x.view(n, m, *x.shape[1:])
        channels = x.size(2)
        graph = graphs[-1].view(n, m, channels, v, v).mean(1).view(n, channels, v * v)
        reconstructed = torch.stack([self.prn(item.t()).t().view(channels, v, v) for item in graph])
        reconstructed = self.relu(self.bn(self.post(reconstructed))).mean(1).view(n, -1)
        self.view_logits = torch.stack(view_logits_list, dim=0).mean(dim=0)
        self.view_logits = self.view_logits.view(n, m, -1).mean(dim=1)
        return x, reconstructed
