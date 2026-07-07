import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..graph import Graph

logger = logging.getLogger(__name__)

EPS = 1e-4


def _shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (tuple, list)):
        return [_shape(i) for i in x]
    return type(x).__name__


def _apply_activation(x, act_name):
    act_name = (act_name or '').lower()
    if act_name == 'tanh':
        return torch.tanh(x)
    if act_name == 'sigmoid':
        return torch.sigmoid(x)
    if act_name == 'softmax':
        return torch.softmax(x, dim=-1)
    return F.relu(x)


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        num_joints=25,
        dropout=0.0,
        ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'),
        stride=1,
    ):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU(inplace=True)
        self.num_joints = num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1))
                )
                continue
            if isinstance(cfg, (tuple, list)) and cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
                continue
            if isinstance(cfg, (tuple, list)):
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        UnitTCN(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], dropout=dropout),
                    )
                )
                continue
            raise ValueError(f'Unsupported mstcn branch config: {cfg!r}')

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels),
            self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        n, c, t, v = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :v]
        global_feat = out[..., v]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:v])
        feat = local_feat + global_feat
        return self.transform(feat)

    def forward(self, x):
        out = self.inner_forward(x)
        return self.drop(self.bn(out))


class UnitGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        use_view_branch=True,
        view_num=11,
        ratio=0.125,
        intra_act='softmax',
        inter_act='tanh',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.use_view_branch = use_view_branch
        self.view_num = view_num
        self.ratio = ratio
        self.mid_channels = max(1, int(ratio * out_channels))

        self.intra_act = intra_act
        self.inter_act = inter_act

        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels * self.num_subsets),
            nn.ReLU(inplace=True),
        )
        self.post = nn.Conv2d(self.mid_channels * self.num_subsets, out_channels, 1, bias=False)
        if self.use_view_branch:
            self.view_conv = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1, bias=False)
            self.view_gap = nn.AdaptiveAvgPool2d(1)
            self.view_fc = nn.Linear(self.mid_channels * self.num_subsets, view_num)
            self.view_softmax = nn.Softmax(dim=-1)
            self.view_mats = nn.Parameter(A.clone().unsqueeze(0).repeat(view_num, 1, 1, 1))

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x)

        A = self.A[None, :, None, None]
        if self.use_view_branch:
            view_feat = self.view_gap(self.view_conv(x)).view(n, -1)
            view_logits = self.view_fc(view_feat)
            view_prob = self.view_softmax(view_logits)
            self.last_view_logits = view_logits
            self.last_view_prob = view_prob

            A_view = torch.einsum('nv,vkxy->nkxy', view_prob, self.view_mats)
            A_view = A_view[:, :, None, None]
            A = (A + A_view) / 2
        else:
            self.last_view_logits = None
            self.last_view_prob = None

        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(dim=-2, keepdim=True)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(dim=-2, keepdim=True)

        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        inter_graph = _apply_activation(diff, self.inter_act) * self.alpha[0]
        intra_graph = torch.einsum('nkctv,nkctw->nkctvw', x1, x2)
        intra_graph = _apply_activation(intra_graph, self.intra_act) * self.beta[0]

        A = inter_graph + intra_graph + A
        A = A.squeeze(3)
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
        x = x.reshape(n, -1, t, v)
        x = self.post(x)

        get_gcl_graph = (inter_graph + intra_graph).squeeze(3).reshape(n, -1, v, v)
        return F.relu(self.bn(x) + res), get_gcl_graph


class PrototypeReconstructionNetwork(nn.Module):
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias=False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias=False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.softmax(self.query_matrix(x), dim=-1)
        z = self.memory_matrix(query)
        return self.dropout(z)


class ProtoGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {
            'use_view_branch': kwargs.pop('use_view_branch', True),
            'view_num': kwargs.pop('view_num', 11),
            'ratio': kwargs.pop('gcn_ratio', kwargs.pop('ratio', 0.125)),
            'intra_act': kwargs.pop('gcn_intra_act', kwargs.pop('intra_act', 'softmax')),
            'inter_act': kwargs.pop('gcn_inter_act', kwargs.pop('inter_act', 'tanh')),
        }
        self.gcn = UnitGCN(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = MSTCN(
            out_channels,
            out_channels,
            stride=stride,
            dropout=kwargs.get('tcn_dropout', 0.0),
            ms_cfg=kwargs.get('tcn_ms_cfg', ((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1')),
            num_joints=A.size(-1),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res), gcl_graph


class ProtoGCNBackbone(nn.Module):
    def __init__(
        self,
        graph_cfg,
        in_channels=3,
        base_channels=96,
        ch_ratio=2,
        num_stages=10,
        inflate_stages=(5, 8),
        down_stages=(5, 8),
        data_bn_type='VC',
        num_person=1,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.pretrained = pretrained
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.view_num = kwargs.pop('view_num', 11)
        self.use_view_branch = kwargs.pop('use_view_branch', True)
        self.num_prototype = kwargs.pop('num_prototype', 100)
        self.tcn_ms_cfg = kwargs.pop('tcn_ms_cfg', ((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'))

        lw_kwargs = [dict(kwargs) for _ in range(num_stages)]
        for k, v in list(kwargs.items()):
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = tuple(inflate_stages)
        self.down_stages = tuple(down_stages)

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                ProtoGCNBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    use_view_branch=self.use_view_branch,
                    view_num=self.view_num,
                    tcn_ms_cfg=self.tcn_ms_cfg,
                    **lw_kwargs[0],
                )
            ]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in self.down_stages)
            in_ch = base_channels
            if i in self.inflate_stages:
                inflate_times += 1
            out_ch = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_ch
            modules.append(
                ProtoGCNBlock(
                    in_ch,
                    out_ch,
                    A.clone(),
                    stride,
                    use_view_branch=self.use_view_branch,
                    view_num=self.view_num,
                    tcn_ms_cfg=self.tcn_ms_cfg,
                    **lw_kwargs[i - 1],
                )
            )

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

        out_channels = base_channels
        self.graph_channels = self.gcn[-1].gcn.mid_channels * self.gcn[-1].gcn.num_subsets
        self.post = nn.Conv2d(self.graph_channels, self.graph_channels, 1)
        self.bn = nn.BatchNorm2d(self.graph_channels)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        self.prn = PrototypeReconstructionNetwork(self.graph_channels, self.num_prototype)
        self.view_logits = None

    def init_weights(self):
        if isinstance(self.pretrained, str) and self.pretrained and os.path.exists(self.pretrained):
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logger.info('Loaded ProtoGCN pretrained weights from %s', self.pretrained)
            if missing:
                logger.info('Missing keys: %s', missing)
            if unexpected:
                logger.info('Unexpected keys: %s', unexpected)

    def forward(self, x):
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(n, m * v * c, t))
        else:
            x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        get_graph = []
        view_logits_list = []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            get_graph.append(gcl_graph)
            view_logits = getattr(self.gcn[i].gcn, 'last_view_logits', None)
            if view_logits is not None:
                view_logits_list.append(view_logits)

        x = x.reshape((n, m) + x.shape[1:])
        c_graph = get_graph[-1].size(1)

        graph = get_graph[-1]
        graph = graph.view(n, m, c_graph, v, v).mean(1).view(n, c_graph, v * v)

        # Run PRN on the whole batch at once; the original per-sample Python loop
        # becomes a major bottleneck in FastPoseGait training.
        re_graph = graph.permute(0, 2, 1).contiguous()
        re_graph = self.prn(re_graph)
        re_graph = re_graph.permute(0, 2, 1).contiguous().view(n, c_graph, v, v)
        re_graph = self.post(re_graph)
        reconstructed_graph = self.relu(self.bn(re_graph))
        reconstructed_graph = reconstructed_graph.mean(1).view(n, -1)

        if self.gcn[0].gcn.use_view_branch and len(view_logits_list) > 0:
            view_logits = torch.stack(view_logits_list, dim=0).mean(dim=0)
            view_logits = view_logits.view(n, m, -1).mean(dim=1)
            self.view_logits = view_logits
        else:
            self.view_logits = None

        return x, reconstructed_graph


class ProtoGCN(BaseModel):
    def build_network(self, model_cfg):
        graph_cfg = model_cfg.get('graph_cfg', {})
        if 'joint_format' in model_cfg and 'joint_format' not in graph_cfg:
            graph_cfg['joint_format'] = model_cfg['joint_format']
        if 'max_hop' not in graph_cfg:
            graph_cfg['max_hop'] = model_cfg.get('max_hop', 3)

        self.view_num = model_cfg.get('view_num', 11)
        self.use_view_branch = model_cfg.get('use_view_branch', True)
        self.view_loss_weight = model_cfg.get('view_loss_weight', 1.0)
        self.enable_visual_summary = model_cfg.get('enable_visual_summary', False)
        self.num_class = model_cfg['num_class']
        self.dropout = nn.Dropout(model_cfg.get('dropout', 0.0))

        self.encoder = ProtoGCNBackbone(
            graph_cfg=graph_cfg,
            in_channels=model_cfg.get('in_channels', 3),
            base_channels=model_cfg.get('base_channels', 96),
            ch_ratio=model_cfg.get('ch_ratio', 2),
            num_stages=model_cfg.get('num_stages', 10),
            inflate_stages=tuple(model_cfg.get('inflate_stages', (5, 8))),
            down_stages=tuple(model_cfg.get('down_stages', (5, 8))),
            data_bn_type=model_cfg.get('data_bn_type', 'VC'),
            num_person=model_cfg.get('num_person', 1),
            pretrained=model_cfg.get('pretrained', None),
            use_view_branch=self.use_view_branch,
            view_num=self.view_num,
            num_prototype=model_cfg.get('num_prototype', 300),
            tcn_ms_cfg=model_cfg.get('tcn_ms_cfg', ((3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1')),
            tcn_dropout=model_cfg.get('tcn_dropout', 0.0),
            gcn_ratio=model_cfg.get('gcn_ratio', 0.125),
            gcn_intra_act=model_cfg.get('gcn_intra_act', 'softmax'),
            gcn_inter_act=model_cfg.get('gcn_inter_act', 'tanh'),
        )
        self.classifier = nn.Linear(self.encoder.out_channels, self.num_class)

    def init_parameters(self):
        super().init_parameters()
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()

    @staticmethod
    def _pool_features_before_head(x):
        if isinstance(x, (tuple, list)):
            x = torch.cat(x, dim=2)
        if len(x.shape) == 2:
            return x
        if len(x.shape) != 5:
            raise ValueError(f'Unsupported feature shape for extraction: {tuple(x.shape)}')
        pool = nn.AdaptiveAvgPool2d(1)
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v)
        x = pool(x)
        x = x.reshape(n, m, c)
        return x.mean(dim=1)

    @staticmethod
    def _unwrap_meta(meta):
        if meta is None:
            return None
        if hasattr(meta, 'data'):
            return ProtoGCN._unwrap_meta(meta.data)
        if isinstance(meta, (list, tuple)):
            return [ProtoGCN._unwrap_meta(item) for item in meta]
        return meta

    def _view_to_index(self, view):
        if isinstance(view, torch.Tensor):
            view = view.detach().cpu().view(-1)[0].item()
        elif isinstance(view, np.ndarray):
            view = np.asarray(view).reshape(-1)[0].item()

        if isinstance(view, str):
            view = view.strip()
            if view.isdigit():
                view = int(view)
            else:
                view = int(float(view))

        view = int(view)
        if 0 <= view < self.view_num:
            return view
        if 1 <= view <= self.view_num:
            return view - 1
        if 0 <= view <= 180 and view % 18 == 0:
            return view // 18
        raise ValueError(f'Unsupported view value: {view}')

    def _extract_view_labels(self, views, device):
        views = self._unwrap_meta(views)
        if views is None:
            return None
        if isinstance(views, dict):
            views = [views]
        elif not isinstance(views, (list, tuple)):
            views = [views]
        labels = [self._view_to_index(v) for v in views]
        return torch.tensor(labels, device=device, dtype=torch.long)

    def _normalize_input(self, pose):
        if pose.dim() == 4:
            pose = pose.unsqueeze(-1)
        if pose.dim() != 5:
            raise ValueError(f'Unsupported pose shape: {tuple(pose.shape)}')
        if pose.shape[-1] == 1 or pose.shape[1] >= pose.shape[-1]:
            pose = pose.permute(0, 4, 2, 3, 1).contiguous()
        else:
            raise ValueError(f'Unexpected pose layout: {tuple(pose.shape)}')
        return pose

    def forward(self, inputs):
        ipts, labs, _, views, seqL = inputs

        pose = ipts[0]
        pose = self._normalize_input(pose)
        n, m, t, v, c = pose.size()

        backbone_feat, reconstructed_graph = self.encoder(pose)
        pooled_feat = self._pool_features_before_head(backbone_feat)
        pooled_feat = self.dropout(pooled_feat)
        logits = self.classifier(pooled_feat)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': pooled_feat.unsqueeze(-1), 'labels': labs},
                'softmax': {'logits': logits.unsqueeze(-1), 'labels': labs},
                'csc': {
                    'features': reconstructed_graph,
                    'labels': labs,
                    'logits': logits,
                },
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': pooled_feat.unsqueeze(-1),
            },
        }
        if self.enable_visual_summary:
            retval['visual_summary']['image/pose'] = pose.view(n * t, m, v, c).contiguous()

        view_logits = getattr(self.encoder, 'view_logits', None)
        if self.use_view_branch and view_logits is not None:
            view_labels = self._extract_view_labels(views, device=labs.device)
            if view_labels is None:
                raise ValueError('View logits are available but view labels are missing.')
            retval['training_feat']['view_softmax'] = {
                'logits': view_logits.unsqueeze(-1),
                'labels': view_labels,
            }

        return retval
