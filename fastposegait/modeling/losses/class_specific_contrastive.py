import torch
import torch.nn as nn

from .base import BaseLoss, gather_and_scale_wrapper


class ClassSpecificContrastiveLoss(BaseLoss):
    def __init__(self, n_class, n_channel=289, h_channel=256, tmp=0.125, mom=0.9, pred_threshold=0.0, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.n_class = n_class
        self.tmp = tmp
        self.mom = mom
        self.pred_threshold = pred_threshold
        self.register_buffer('avg_f', torch.randn(self.h_channel, self.n_class))
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def onehot(self, label):
        lbl = label.clone().view(-1)
        ones = torch.eye(self.n_class, device=label.device)
        return ones.index_select(0, lbl.long()).float()

    def get_mask(self, lbl_one, pred_one, logit):
        tp = lbl_one * pred_one
        tp = tp * (logit > self.pred_threshold).float()
        return tp

    def local_average(self, f, mask):
        f = f.permute(1, 0)
        avg_f = self.avg_f.detach().to(f.device)
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask)
        f_mask = f_mask / (mask_sum + 1e-12)

        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem
        return f_mem

    def get_score(self, feature, f_mem):
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)
        f_mem = f_mem.permute(1, 0)
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)
        score_mem = torch.matmul(f_mem, feature.permute(1, 0))
        score_cl = score_mem / self.tmp
        return score_cl

    @gather_and_scale_wrapper
    def forward(self, feature, lbl, logit):
        feature = self.cl_fc(feature)
        lbl = lbl.view(-1)
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)
        logit = torch.softmax(logit, 1)

        mask = self.get_mask(lbl_one, pred_one, logit)
        f_mem = self.local_average(feature, mask)
        score_cl = self.get_score(feature, f_mem)
        score_cl = score_cl.permute(1, 0).contiguous()

        loss = self.loss(score_cl, lbl).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
