import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..graph import Graph

EPS = 1e-4


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
        num_joints=17,
        dropout=0.0,
        ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
        stride=1,
    ):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
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
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            if cfg[0] == "max":
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
                continue
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    UnitTCN(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], dropout=dropout),
                )
            )
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

        branch_outs = []
        for tempconv in self.branches:
            branch_outs.append(tempconv(x))

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :v]
        global_feat = out[..., v]
        global_feat = torch.einsum("nct,v->nctv", global_feat, self.add_coeff[:v])
        feat = local_feat + global_feat
        return self.transform(feat)

    def forward(self, x):
        out = self.inner_forward(x)
        return self.drop(self.bn(out))


class ProtoGCNUnitGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        view_num=11,
        ratio=0.125,
        intra_act="softmax",
        inter_act="tanh",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.view_num = view_num
        self.ratio = ratio
        self.mid_channels = max(1, int(ratio * out_channels))

        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1),
            nn.BatchNorm2d(self.mid_channels * self.num_subsets),
            nn.ReLU(),
        )
        self.post = nn.Conv2d(self.mid_channels * self.num_subsets, out_channels, 1)
        self.view_conv = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)
        self.view_gap = nn.AdaptiveAvgPool2d(1)
        self.view_fc = nn.Linear(self.mid_channels * self.num_subsets, view_num)
        self.view_softmax = nn.Softmax(dim=-1)
        self.view_mats = nn.Parameter(A.clone().unsqueeze(0).repeat(view_num, 1, 1, 1))

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.intra_act = intra_act
        self.inter_act = inter_act

    def _apply_act(self, name, x):
        if name == "softmax":
            return self.softmax(x)
        if name == "tanh":
            return self.tanh(x)
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x)

        view_feat = self.view_gap(self.view_conv(x)).view(n, -1)
        view_logits = self.view_fc(view_feat)
        view_prob = self.view_softmax(view_logits)
        self.last_view_logits = view_logits
        self.last_view_prob = view_prob

        # Base adjacency plus view-conditioned adaptive adjacency.
        A = self.A.unsqueeze(0)
        A_view = torch.einsum("nv,vkxy->nkxy", view_prob, self.view_mats)
        A = (A + A_view) / 2.0

        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)

        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = x1.mean(dim=3)
        x2 = x2.mean(dim=3)

        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        inter_graph = self._apply_act(self.inter_act, diff).mean(dim=2) * self.alpha[0]
        intra_graph = torch.einsum("nkmv,nkmw->nkvw", x1, x2)
        intra_graph = self._apply_act(self.intra_act, intra_graph) * self.beta[0]

        A = A + inter_graph + intra_graph
        x = torch.einsum("nkctv,nkvw->nkctw", pre_x, A).contiguous()
        x = x.reshape(n, -1, t, v)
        gcl_graph = inter_graph + intra_graph
        gcl_graph = gcl_graph.reshape(n, -1, v, v)
        return self.relu(self.bn(self.post(x)) + res), gcl_graph


class ProtoGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, view_num=11, kernel_size=(9, 2)):
        super().__init__()
        temporal_window_size, _ = kernel_size
        self.gcn = ProtoGCNUnitGCN(in_channels, out_channels, A, view_num=view_num)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride, num_joints=A.size(-1))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, (stride, 1)), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()

    def forward(self, x, A=None):
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res), gcl_graph


class ProtoGCNTriplet(BaseModel):
    def build_network(self, model_cfg):
        self.joint_format = model_cfg.get("joint_format", "coco")
        self.in_channels = model_cfg.get("in_channels", 10)
        self.base_channels = model_cfg.get("base_channels", 64)
        self.ch_ratio = model_cfg.get("ch_ratio", 2)
        self.num_stages = model_cfg.get("num_stages", 10)
        self.inflate_stages = set(model_cfg.get("inflate_stages", [5, 8]))
        self.down_stages = set(model_cfg.get("down_stages", [5, 8]))
        self.view_num = model_cfg.get("view_num", 11)
        self.embed_dim = model_cfg.get("embed_dim", 256)
        self.max_hop = model_cfg.get("max_hop", 2)

        self.graph = Graph(joint_format=self.joint_format, max_hop=self.max_hop)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.data_bn = nn.BatchNorm1d(self.in_channels * A.size(1))

        modules = []
        current_channels = self.base_channels
        if self.in_channels != current_channels:
            modules.append(
                ProtoGCNBlock(
                    self.in_channels,
                    current_channels,
                    A.clone(),
                    stride=1,
                    residual=False,
                    view_num=self.view_num,
                )
            )

        for stage_idx in range(2, self.num_stages + 1):
            stride = 2 if stage_idx in self.down_stages else 1
            out_channels = current_channels * (self.ch_ratio if stage_idx in self.inflate_stages else 1)
            out_channels = int(out_channels + EPS)
            modules.append(
                ProtoGCNBlock(
                    current_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    view_num=self.view_num,
                )
            )
            current_channels = out_channels

        self.backbone = nn.ModuleList(modules)
        self.out_channels = current_channels
        self.embed_proj = nn.Conv1d(self.out_channels, self.embed_dim, kernel_size=1)

    def _reshape_input(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        if x.dim() != 5:
            raise ValueError(f"Expected input shape [N, C, T, V, M], got {tuple(x.shape)}")
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        return x, n, m

    def extract_feat(self, x):
        x, n, m = self._reshape_input(x)
        last_graph = None
        for block in self.backbone:
            x, last_graph = block(x)

        x = x.view(n, m, x.size(1), x.size(2), x.size(3))
        x = x.mean(dim=1)  # person pooling
        x = x.mean(dim=2)  # temporal pooling -> [N, C, V]
        x = self.embed_proj(x)
        return x, last_graph

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        pose = ipts[0]

        feat, last_graph = self.extract_feat(pose)
        embed = feat

        retval = {
            "training_feat": {
                "triplet": {"embeddings": embed, "labels": labs},
            },
            "visual_summary": {},
            "inference_feat": {
                "embeddings": embed,
            },
        }
        self.last_graph = last_graph
        return retval
