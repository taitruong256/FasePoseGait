"""Gait-recognition model using the pure-PyTorch ProtoGCN backbone."""
import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..backbones.protogcn_backbone import ProtoGCNBackbone
from ..losses.class_specific_contrastive import ClassSpecificContrastiveLoss
from utils import get_valid_args


class ProtoGCN(BaseModel):
    def build_network(self, model_cfg):
        self.num_class = model_cfg['num_class']
        self.log_pose = model_cfg.get('log_pose', False)
        self.backbone = ProtoGCNBackbone(**model_cfg['backbone_cfg'])
        embedding_dim = model_cfg.get('embedding_dim', self.backbone.out_channels)
        self.embedding_proj = (torch.nn.Linear(self.backbone.out_channels, embedding_dim)
                               if embedding_dim != self.backbone.out_channels else torch.nn.Identity())
        # Equivalent to ProtoGCN's SimpleHead.fc_cls.
        self.fc_cls = torch.nn.Linear(self.backbone.out_channels, self.num_class)
        csc_cfg = next(
            (cfg for cfg in self.cfgs['loss_cfg']
             if cfg['type'] == 'ClassSpecificContrastiveLoss'), None)
        if csc_cfg is None or not csc_cfg.get('model_managed', False):
            raise ValueError(
                'ProtoGCN requires a model_managed ClassSpecificContrastiveLoss in loss_cfg.')
        csc_args = get_valid_args(
            ClassSpecificContrastiveLoss, csc_cfg,
            free_keys=['type', 'log_prefix', 'model_managed'])
        self.csc_loss = ClassSpecificContrastiveLoss(**csc_args)

    def forward(self, inputs):
        ipts, labels, _, _, _ = inputs
        pose = ipts[0]
        n, c, t, v, m = pose.shape
        features, reconstructed_graph = self.backbone(pose.permute(0, 4, 2, 3, 1).contiguous())
        pooled_features = features.mean(dim=(1, 3, 4))
        cls_score = self.fc_cls(pooled_features)
        embeddings = F.normalize(self.embedding_proj(pooled_features), p=2, dim=1).unsqueeze(-1)
        if self.training:
            # Match ProtoGCN BaseHead.loss: CSC only updates a class memory
            # when the detached classifier prediction is correct/confident.
            csc_loss, _ = self.csc_loss(
                features=reconstructed_graph, labels=labels, logits=cls_score.detach())
            csc_loss = csc_loss * self.csc_loss.loss_term_weight
        else:
            # Evaluation must not update CSC's class-memory buffer.
            csc_loss = reconstructed_graph.new_zeros(())
        visual_summary = {}
        if self.log_pose:
            visual_summary['image/pose'] = (
                pose.permute(0, 2, 4, 3, 1).contiguous().view(n * t, m, v, c))

        return {
            'training_feat': {
                # CrossEntropyLoss is configured with scale=1 and no label
                # smoothing, which is standard CE as used by ProtoGCN.
                'cross_entropy': {'logits': cls_score.unsqueeze(-1), 'labels': labels},
                'csc': csc_loss,
            },
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embeddings},
        }
