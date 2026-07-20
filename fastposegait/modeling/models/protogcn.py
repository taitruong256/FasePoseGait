"""Gait-recognition model using the pure-PyTorch ProtoGCN backbone."""
import torch
import torch.nn.functional as F
import logging

from ..base_model import BaseModel
from ..backbones.protogcn_backbone import ProtoGCNBackbone
from ..losses.class_specific_contrastive import ClassSpecificContrastiveLoss
from utils import get_valid_args

logger = logging.getLogger(__name__)


class ProtoGCN(BaseModel):
    def build_network(self, model_cfg):
        self.num_class = model_cfg['num_class']
        self.log_pose = model_cfg.get('log_pose', False)
        self.view_loss_weight = model_cfg.get('view_loss_weight', 0.)
        self.view_step = model_cfg.get('view_step', 18)
        self.random_rotation_theta = model_cfg.get('random_rotation_theta', 0.)
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

    def init_parameters(self):
        super().init_parameters()
        # ProtoGCN SimpleHead initializes its classifier with std=0.01.
        torch.nn.init.normal_(self.fc_cls.weight, std=0.01)
        torch.nn.init.constant_(self.fc_cls.bias, 0.)

    def _view_targets(self, views, device):
        try:
            targets = [int(str(view)) // self.view_step for view in views]
        except ValueError as error:
            raise ValueError('CASIA-B views must be numeric strings, got {}.'.format(views)) from error
        targets = torch.tensor(targets, device=device, dtype=torch.long)
        if targets.min() < 0 or targets.max() >= self.backbone.gcn[0].gcn.view_fc.out_features:
            raise ValueError('View target is outside ProtoGCN view classes: {}.'.format(targets.tolist()))
        return targets

    def _random_rotate(self, pose):
        """Apply one 2D rotation per sequence, matching ProtoGCN RandomRot."""
        if not self.training or not self.random_rotation_theta:
            return pose
        angles = pose.new_empty(pose.size(0)).uniform_(
            -self.random_rotation_theta, self.random_rotation_theta)
        cos, sin = torch.cos(angles)[:, None, None, None], torch.sin(angles)[:, None, None, None]
        rotated = pose.clone()
        x, y = pose[:, 0], pose[:, 1]
        rotated[:, 0] = cos * x - sin * y
        rotated[:, 1] = sin * x + cos * y
        return rotated

    def forward(self, inputs):
        ipts, labels, _, views, seqL = inputs
        pose = ipts[0]
        
        if self.training:
            logger.debug(f"[ProtoGCN.forward] Training mode, input shape: {pose.shape}, labels: {labels.shape}")
        else:
            logger.debug(f"[ProtoGCN.forward] Inference mode, input shape: {pose.shape}, labels: {labels.shape}")
        
        pose = self._random_rotate(pose)
        n, c, t, v, m = pose.shape
        logger.debug(f"[ProtoGCN.forward] After rotation: {pose.shape}")
        
        if seqL is None:
            features, reconstructed_graph = self.backbone(
                pose.permute(0, 4, 2, 3, 1).contiguous())
            logger.debug(f"[ProtoGCN.forward] Backbone output - features: {features.shape}, reconstructed_graph: {reconstructed_graph.shape}")
            pooled_features = features.mean(dim=(1, 3, 4))
            logger.debug(f"[ProtoGCN.forward] After global pooling: {pooled_features.shape}")
        else:
            lengths = seqL.reshape(-1).detach().cpu().tolist()
            if sum(lengths) != t:
                raise ValueError(
                    'Packed sequence lengths ({}) do not match input frames ({}).'.format(
                        sum(lengths), t))
            pooled = []
            start = 0
            for length in lengths:
                sequence_pose = pose[:, :, start:start + length]
                features, _ = self.backbone(
                    sequence_pose.permute(0, 4, 2, 3, 1).contiguous())
                pooled.append(features.mean(dim=(1, 3, 4)))
                start += length
            pooled_features = torch.cat(pooled, dim=0)
            reconstructed_graph = None
            logger.debug(f"[ProtoGCN.forward] Packed sequence mode - pooled_features: {pooled_features.shape}")
        
        cls_score = self.fc_cls(pooled_features)
        logger.debug(f"[ProtoGCN.forward] Classifier output: {cls_score.shape}")
        
        embeddings = F.normalize(self.embedding_proj(pooled_features), p=2, dim=1).unsqueeze(-1)
        logger.debug(f"[ProtoGCN.forward] Embeddings: {embeddings.shape}")
        
        if self.training:
            if reconstructed_graph is None:
                raise ValueError('Packed sequences are only supported during inference.')
            # Match ProtoGCN BaseHead.loss: CSC only updates a class memory
            # when the detached classifier prediction is correct/confident.
            csc_loss, _ = self.csc_loss(
                features=reconstructed_graph, labels=labels, logits=cls_score.detach())
            csc_loss = csc_loss * self.csc_loss.loss_term_weight
            if self.view_loss_weight:
                view_targets = self._view_targets(views, cls_score.device)
                view_loss = F.cross_entropy(self.backbone.view_logits, view_targets)
                view_loss = view_loss * self.view_loss_weight
            else:
                view_loss = cls_score.new_zeros(())
        else:
            # Evaluation must not update CSC's class-memory buffer.
            csc_loss = cls_score.new_zeros(())
            view_loss = cls_score.new_zeros(())
        
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
                'view': view_loss,
            },
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embeddings},
        }
