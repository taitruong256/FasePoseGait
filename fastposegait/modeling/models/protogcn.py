"""Gait-recognition model using the pure-PyTorch ProtoGCN backbone."""
import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..backbones.protogcn_backbone import ProtoGCNBackbone
from ..losses.class_specific_contrastive import ClassSpecificContrastiveLoss


class ProtoGCN(BaseModel):
    def build_network(self, model_cfg):
        self.num_class = model_cfg['num_class']
        self.backbone = ProtoGCNBackbone(**model_cfg['backbone_cfg'])
        embedding_dim = model_cfg.get('embedding_dim', self.backbone.out_channels)
        self.embedding_proj = (torch.nn.Linear(self.backbone.out_channels, embedding_dim)
                               if embedding_dim != self.backbone.out_channels else torch.nn.Identity())
        self.csc_loss = ClassSpecificContrastiveLoss(**model_cfg['csc_loss_cfg'])

    def forward(self, inputs):
        ipts, labels, _, _, _ = inputs
        pose = ipts[0]
        n, c, t, v, m = pose.shape
        features, reconstructed_graph = self.backbone(pose.permute(0, 4, 2, 3, 1).contiguous())
        embeddings = features.mean(dim=(1, 3, 4))
        embeddings = F.normalize(self.embedding_proj(embeddings), p=2, dim=1).unsqueeze(-1)
        if self.training:
            csc_loss, _ = self.csc_loss(reconstructed_graph, labels)
            csc_loss = csc_loss * self.csc_loss.loss_term_weight
        else:
            # Evaluation must not update CSC's class-memory buffer.
            csc_loss = reconstructed_graph.new_zeros(())
        return {
            'training_feat': {
                'triplet': {'embeddings': embeddings, 'labels': labels},
                'csc': csc_loss,
            },
            'visual_summary': {'image/pose': pose.permute(0, 2, 4, 3, 1).contiguous().view(n * t, m, v, c)},
            'inference_feat': {'embeddings': embeddings},
        }
