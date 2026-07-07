import torch
import torch.nn as nn

from .base import BaseLoss


class ClassSpecificContrastiveLoss(BaseLoss):
    def __init__(
        self,
        num_classes,
        feature_channels,
        hidden_channels=256,
        temperature=0.125,
        momentum=0.9,
        pred_threshold=0.0,
        loss_term_weight=1.0,
    ):
        super().__init__(loss_term_weight)
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.temperature = temperature
        self.momentum = momentum
        self.pred_threshold = pred_threshold

        self.cl_fc = nn.Linear(feature_channels, hidden_channels)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.register_buffer('avg_f', torch.randn(hidden_channels, num_classes))

    def onehot(self, labels):
        labels = labels.view(-1)
        return torch.eye(self.num_classes, device=labels.device).index_select(0, labels.long()).float()

    def get_mask(self, lbl_one, pred_one, logit_prob):
        tp = lbl_one * pred_one
        tp = tp * (logit_prob > self.pred_threshold).float()
        return tp

    def local_average(self, features, mask):
        features = features.permute(1, 0)
        avg_f = self.avg_f.detach().to(features.device)

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(features, mask)
        f_mask = f_mask / (mask_sum + 1e-12)

        has_object = (mask_sum > 1e-8).float()
        has_object = torch.where(has_object > 0.1, torch.full_like(has_object, self.momentum), torch.ones_like(has_object))
        f_mem = avg_f * has_object + (1 - has_object) * f_mask

        with torch.no_grad():
            self.avg_f.copy_(f_mem)
        return f_mem

    def get_score(self, features, f_mem):
        features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-12)
        f_mem = f_mem.permute(1, 0)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)
        score_mem = torch.matmul(f_mem, features.permute(1, 0))
        return score_mem / self.temperature

    def forward(self, features, labels, logits):
        features = self.cl_fc(features)
        pred = logits.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(labels)
        logit_prob = torch.softmax(logits, 1)

        mask = self.get_mask(lbl_one, pred_one, logit_prob)
        f_mem = self.local_average(features, mask)
        score_cl = self.get_score(features, f_mem).permute(1, 0).contiguous()
        loss = self.loss(score_cl, labels).mean()

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
