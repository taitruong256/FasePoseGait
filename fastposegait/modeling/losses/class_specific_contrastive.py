"""Class-specific contrastive loss used by the original ProtoGCN GaitHead."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class ClassSpecificContrastiveLoss(BaseLoss):
    def __init__(self, num_classes, input_dim, hidden_dim=256, temperature=.125,
                 momentum=.9, pred_threshold=0., loss_term_weight=1.):
        super().__init__(loss_term_weight)
        self.num_classes, self.temperature = num_classes, temperature
        self.momentum, self.pred_threshold = momentum, pred_threshold
        self.cl_fc = nn.Linear(input_dim, hidden_dim)
        self.register_buffer('avg_f', torch.randn(hidden_dim, num_classes))

    @gather_and_scale_wrapper
    def forward(self, features, labels):
        features = self.cl_fc(features)
        labels = labels.view(-1).long()
        if labels.numel() == 0 or labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                'CSC labels must be in [0, {}], got [{}, {}].'.format(
                    self.num_classes - 1, labels.min().item(), labels.max().item()))
        onehot = F.one_hot(labels, self.num_classes).float()
        mask = onehot * (onehot > self.pred_threshold).float()
        mask_sum = mask.sum(0, keepdim=True)
        batch_means = features.t().matmul(mask) / (mask_sum + 1e-12)
        update_weight = torch.where(mask_sum > 1e-8,
                                    torch.full_like(mask_sum, self.momentum),
                                    torch.ones_like(mask_sum))
        with torch.no_grad():
            self.avg_f.copy_(self.avg_f * update_weight + (1 - update_weight) * batch_means)
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_memory = F.normalize(self.avg_f.t(), p=2, dim=1)
        logits = normalized_features.matmul(normalized_memory.t()) / self.temperature
        loss = F.cross_entropy(logits, labels)
        self.info.update({'loss': loss.detach()})
        return loss, self.info
