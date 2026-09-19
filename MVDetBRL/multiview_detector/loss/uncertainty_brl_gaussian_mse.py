import torch
from torch import nn
import torch.nn.functional as F


class UncertaintyBRLGaussianMSE(nn.Module):
    """
    BRL (Background Recalibration Loss) cho heatmap + modeling noise/uncertainty
    theo tinh than Eq.(6) cua "Open-Vocabulary Instance Segmentation via Robust
    Cross-Modal Pseudo-Labeling" (XPM, CVPR'22) va Kendall & Gal (NeurIPS'17).

    Model phai tra ve them mot nhanh `logvar` (cung shape voi prediction `x`),
    la log cua variance sigma^2 du doan cho tung pixel.

    3 mode:
      - mode='nll'  (KHUYEN DUNG): dang closed-form cua gia thiet
            y ~ N(x, sigma^2)  =>  -log p(y|x,sigma) = (x-y)^2/(2 sigma^2) + 0.5*log sigma^2
        Day chinh la "prediction co nhieu Gaussian" ma thay mo ta, nhung ky vong
        duoc tinh giai tich thay vi Monte-Carlo. No co loss attenuation that su:
        pixel nao model khong chac -> sigma lon -> gradient bi chia cho sigma^2
        -> outlier / nhan bi drop khong pha hong viec hoc.

      - mode='mc'   (bam sat cong thuc trong paper, DE ABLATION):
            se = E_eps[(x + sigma*eps - y)^2],  eps ~ N(0, 1)
        CANH BAO: voi MSE thi E[(x + sigma*eps - y)^2] = (x-y)^2 + sigma^2, nen
        gradient theo sigma luon duong => sigma bi ep ve 0 (collapse). Mode nay
        chi de bao cao ablation / chung minh trong bao cao, khong dung de train.

      - mode='none': tat uncertainty, quay ve BRL goc.

    Neu `logvar=None` thi loss tu dong quay ve BRL goc (dung cho warm-up epochs).

    Luu y: NLL co the AM (khi sigma rat nho). Do la binh thuong voi likelihood
    loss, dung hoang mang khi thay loss < 0; nen log them raw-MSE de theo doi.
    """

    def __init__(self,
                 pos_thr=0.1,
                 confuse_pred_thr=0.3,
                 beta=0.1,
                 mirror=True,
                 mode='nll',
                 apply_on=('pos', 'confuse'),
                 logvar_min=-8.0,
                 logvar_max=2.0,
                 lambda_reg=0.0,
                 num_samples=4):
        super().__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror
        assert mode in ('nll', 'mc', 'none')
        self.mode = mode
        self.apply_on = tuple(apply_on)          # subset cua ('pos','confuse','neg')
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.lambda_reg = lambda_reg             # phat mean(logvar) de chan sigma phi ma
        self.num_samples = num_samples

        # thong ke de log / visualize, khong tham gia graph
        self.last_stats = {}

    # ------------------------------------------------------------------ #
    def forward(self, x, target, kernel, logvar=None):
        soft_gt = self._traget_transform(x, target, kernel)

        pos_mask = soft_gt >= self.pos_thr
        bg_mask = ~pos_mask
        # viec phan vung khong duoc backprop
        confuse_mask = bg_mask & (x.detach() >= self.confuse_pred_thr)
        easy_neg_mask = bg_mask & ~confuse_mask

        if logvar is None or self.mode == 'none':
            lv = None
        else:
            lv = logvar.clamp(self.logvar_min, self.logvar_max)

        loss = x.new_zeros(())
        count = x.new_zeros(())

        if pos_mask.any():
            loss = loss + self._term(x, soft_gt, lv, pos_mask,
                                     weight=1.0, use_unc=('pos' in self.apply_on))
            count = count + pos_mask.sum()

        if easy_neg_mask.any():
            loss = loss + self._term(x, soft_gt, lv, easy_neg_mask,
                                     weight=1.0, use_unc=('neg' in self.apply_on))
            count = count + easy_neg_mask.sum()

        if confuse_mask.any():
            tgt = torch.ones_like(soft_gt) if self.mirror else soft_gt
            loss = loss + self._term(x, tgt, lv, confuse_mask,
                                     weight=self.beta,
                                     use_unc=('confuse' in self.apply_on))
            count = count + confuse_mask.sum()

        loss = loss / count.clamp(min=1).float()

        if lv is not None and self.lambda_reg > 0:
            loss = loss + self.lambda_reg * lv.mean()

        with torch.no_grad():
            self.last_stats = {
                'n_pos': pos_mask.sum().item(),
                'n_confuse': confuse_mask.sum().item(),
                'raw_mse': F.mse_loss(x.detach(), soft_gt).item(),
            }
            if lv is not None:
                sigma = torch.exp(0.5 * lv.detach())
                self.last_stats['sigma_mean'] = sigma.mean().item()
                if confuse_mask.any():
                    self.last_stats['sigma_confuse'] = sigma[confuse_mask].mean().item()
                if pos_mask.any():
                    self.last_stats['sigma_pos'] = sigma[pos_mask].mean().item()

        return loss

    # ------------------------------------------------------------------ #
    def _term(self, x, tgt, lv, mask, weight, use_unc):
        xs, ts = x[mask], tgt[mask]

        if lv is None or not use_unc:
            return weight * F.mse_loss(xs, ts, reduction='sum')

        lvs = lv[mask]

        if self.mode == 'mc':
            sigma = torch.exp(0.5 * lvs)
            eps = torch.randn((self.num_samples,) + xs.shape,
                              device=xs.device, dtype=xs.dtype)
            noisy = xs.unsqueeze(0) + sigma.unsqueeze(0) * eps
            se = ((noisy - ts.unsqueeze(0)) ** 2).mean(0)
            return weight * se.sum()

        # mode == 'nll'
        se = (xs - ts) ** 2
        nll = 0.5 * (se * torch.exp(-lvs) + lvs)
        return weight * nll.sum()

    # ------------------------------------------------------------------ #
    def _traget_transform(self, x, target, kernel):
        # giu nguyen ten ham (trainer.py dang goi truc tiep khi visualize)
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return target


class ViewReliability(nn.Module):
    """
    Tuong duong Eq.(7) cua XPM: alpha_v = eta / mean_sigma_v.

    Dung de thay the he so `--alpha` co dinh khi tron per-view loss vao map loss:
    camera nao bi che khuat / anh nhieu -> sigma lon -> trong so nho.

    eta duoc uoc luong bang EMA cua min(mean_sigma) tren cac batch da thay,
    thay vi tinh offline nhu paper.

    LUU Y QUAN TRONG (paper noi ro): KHONG backprop noise head qua so hang
    reweight nay, neu khong model se hoc cach bao "toi rat khong chac" cho moi thu
    de ha loss. => luon detach sigma o day.
    """

    def __init__(self, momentum=0.99, clamp_max=1.0):
        super().__init__()
        self.momentum = momentum
        self.clamp_max = clamp_max
        self.register_buffer('eta', torch.tensor(float('inf')))

    @torch.no_grad()
    def forward(self, logvars):
        """logvars: list cac tensor logvar cua tung view. Tra ve tensor trong so."""
        sig = torch.stack([torch.exp(0.5 * lv.detach()).mean() for lv in logvars])
        batch_min = sig.min()
        if torch.isinf(self.eta):
            self.eta.fill_(batch_min.item())
        else:
            self.eta.mul_(self.momentum).add_((1 - self.momentum) * batch_min)
        w = (self.eta / sig.clamp(min=1e-6)).clamp(max=self.clamp_max)
        return w / w.mean().clamp(min=1e-6)      # giu nguyen scale tong the
