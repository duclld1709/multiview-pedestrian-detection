"""
brl_loss.py (v3 - fix bug #2: cong thuc recalibration bi suy bien tai p=1.
'mirrored_pos' (dung cong thuc giong het loss duong) khien tong loss ->0
khi model du doan TOAN BO la foreground (p=1 khap noi), tao ra nghiem
"gian lan" toan cuc ma optimizer tim ra rat nhanh. Fix: recalibration
chi la GIAM NHE (scale down) loss am chuan, khong thay bang cong thuc
khac di ve 0 tai p=1 - nen p=1 khap noi khong con la nghiem re nua.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BRLLoss(nn.Module):
    IS_LOGIT_BASED = True

    def __init__(self, alpha=0.25, gamma=2.0, conf_thres=0.5,
                 recalib_weight=0.1, warmup_epochs=0, eps=1e-6):
        super().__init__()
        assert 0.0 < conf_thres < 1.0, "conf_thres (t) phai trong (0, 1)"
        self.alpha = alpha
        self.gamma = gamma
        self.t = conf_thres
        self.recalib_weight = recalib_weight
        self.warmup_epochs = warmup_epochs
        self.eps = eps
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _hard_target(self, x, target):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        return (target > 0).float()

    def _traget_transform(self, x, target, kernel):
        return self._hard_target(x, target)

    def forward(self, x, target, kernel=None):
        hard_target = self._hard_target(x, target)
        pos_mask = hard_target
        neg_mask = 1.0 - hard_target

        log_p = F.logsigmoid(x)
        log_1mp = F.logsigmoid(-x)
        p = torch.sigmoid(x.detach()).clamp(self.eps, 1 - self.eps)

        loss_pos = -self.alpha * (1 - p).pow(self.gamma) * log_p

        # normal_neg -> +inf khi p->1 (dung, de "predict tat ca la positive"
        # khong bao gio la nghiem mien phi).
        normal_neg = -self.alpha * p.pow(self.gamma) * log_1mp

        if self.current_epoch <= self.warmup_epochs:
            loss_neg = normal_neg
        else:
            confuse_mask = (p >= (1.0 - self.t)).float()
            # Recalibration = GIAM NHE hinh phat (nhan he so < 1), KHONG
            # thay bang cong thuc khac trieu tieu ve 0 tai p=1.
            recalibrated_neg = self.recalib_weight * normal_neg
            loss_neg = confuse_mask * recalibrated_neg + (1 - confuse_mask) * normal_neg

        loss = loss_pos * pos_mask + loss_neg * neg_mask

        num_pixels = pos_mask.numel()
        num_pos = pos_mask.sum().clamp(min=max(1.0, 0.001 * num_pixels))
        return loss.sum() / num_pos
