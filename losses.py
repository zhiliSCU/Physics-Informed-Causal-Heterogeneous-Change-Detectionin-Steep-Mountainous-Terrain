"""
PhyCDNet Loss Functions.
  L_total = L_CD + λ_phys·L_phys + λ_orth·R_orth + λ_smooth·R_terrain

Three-phase dynamic weight scheduling (Sec. 3.5):
  Phase 1 (0-30%): high λ_phys, enforce physical consistency
  Phase 2 (30-70%): transition, gradually increase λ_orth and λ_smooth
  Phase 3 (70-100%): CD-dominant fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math as _math
import numpy as np


# ============================================================
# 1. Change Detection Loss — Focal + Dice (Tversky)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logit, target):
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1).permute(0, 2, 1).contiguous().view(-1, num_class)
        target = target.view(-1)
        one_hot = torch.zeros(target.size(0), num_class, device=logit.device)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        pt = (one_hot * F.softmax(logit, dim=-1)).sum(1) + self.smooth
        logpt = pt.log()
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=logit.device)[target]
            loss = -alpha * torch.pow(1 - pt, self.gamma) * logpt
        else:
            loss = -torch.pow(1 - pt, self.gamma) * logpt
        return loss.mean()


class DiceLoss(nn.Module):
    """Tversky-based Dice loss with adaptive FP/FN penalty."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha; self.beta = beta
        self.smooth = smooth; self.ignore_index = ignore_index

    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        pred = F.softmax(pred, dim=1)
        num_class = pred.shape[1]
        mask = target != self.ignore_index
        target = target[mask]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, num_class)[mask.view(-1), :]
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        tp = (pred * one_hot).sum(dim=0)
        fp = (pred * (1 - one_hot)).sum(dim=0)
        fn = ((1 - pred) * one_hot).sum(dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class ChangeDetectionLossPhy(nn.Module):
    """Combined Focal + Dice loss."""
    def __init__(self, lambda_f=1.0, lambda_d=1.0, lambda_ms=0.3,
                 focal_alpha=None, dice_alpha=0.7, dice_beta=0.3, ignore_index=255):
        super().__init__()
        self.lambda_f = lambda_f; self.lambda_d = lambda_d; self.lambda_ms = lambda_ms
        self.focal = FocalLoss(alpha=focal_alpha, gamma=2.0)
        self.dice = DiceLoss(alpha=dice_alpha, beta=dice_beta, ignore_index=ignore_index)

    def forward(self, pred, target, aux_preds=None):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        loss_main = self.lambda_f * self.focal(pred, target) + self.lambda_d * self.dice(pred, target)
        ms_loss = 0.0
        for s in [0.5, 0.25]:
            h, w = int(pred.shape[2] * s), int(pred.shape[3] * s)
            sp = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
            st = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()
            ms_loss += self.lambda_f * self.focal(sp, st) + self.lambda_d * self.dice(sp, st)
        return loss_main + self.lambda_ms * (ms_loss / 2)


# ============================================================
# 2. Physical Consistency Loss — L_phys (Sec. 3.2)
# ============================================================
class PhysicalConsistencyLossPhy(nn.Module):
    """L_phys = ||F_path - LF(F_A)||² + ||T_eff - T_ref||²"""
    def __init__(self, kernel_size=11, sigma=3.0, T_ref=0.75):
        super().__init__()
        self.T_ref = T_ref
        C = 256
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.C = C
        self.kernel_size = kernel_size

    def forward(self, F_path_hat, F_A, T_eff_hat):
        with torch.no_grad():
            target_path = F.conv2d(F_A, self.kernel, padding=self.kernel_size//2, groups=self.C)
        loss_path = F.mse_loss(F_path_hat, target_path)
        loss_trans = F.mse_loss(T_eff_hat, torch.full_like(T_eff_hat, self.T_ref))
        return loss_path + loss_trans


# ============================================================
# 3. Orthogonal Disentanglement Loss — R_orth (Sec. 3.3)
# ============================================================
class OrthogonalDisentanglementLoss(nn.Module):
    """R_orth = ||F_terrain^T @ F_change||²_F / (N·C)"""
    def forward(self, F_terrain, F_change, A=None):
        B, C, H, W = F_terrain.shape
        N = B * H * W
        F_t = F_terrain.permute(0, 2, 3, 1).contiguous().view(N, C)
        F_c = F_change.permute(0, 2, 3, 1).contiguous().view(N, C)
        cross = torch.mm(F_t.T, F_c)
        orth_loss = (cross ** 2).sum() / (N * C)
        return {'orth_loss': orth_loss, 'total': orth_loss}


# ============================================================
# 4. Terrain-adaptive Smoothness — R_terrain (Sec. 3.4, Eq. X)
# ============================================================
class TerrainSmoothnessLoss(nn.Module):
    """R_terrain(φ) = Σ_x ||∇φ(x)||² / (||∇h(x)|| + α)"""
    def __init__(self, alpha=1e-3):
        super().__init__()
        self.alpha = alpha

    def forward(self, phi, dem_gradient=None):
        if phi is None:
            return torch.tensor(0.0)
        B, _, H, W = phi.shape
        dx = phi[:, :, :, 1:] - phi[:, :, :, :-1]
        dy = phi[:, :, 1:, :] - phi[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0)); dy = F.pad(dy, (0, 0, 0, 1))
        gm = (dx ** 2 + dy ** 2).sum(dim=1, keepdim=True)
        if dem_gradient is None:
            return gm.mean()
        if dem_gradient.shape[-2:] != (H, W):
            dem_gradient = F.interpolate(dem_gradient, size=(H, W),
                                          mode='bilinear', align_corners=False)
        return (gm / (dem_gradient.abs() + self.alpha)).mean()


# ============================================================
# 5. Dynamic Weight Scheduler (Sec. 3.5)
# ============================================================
class DynamicWeightScheduler:
    """Three-phase dynamic scheduling with cosine interpolation."""
    def __init__(self, total_epochs, lambda1_range=(3.0, 0.3),
                 lambda2_range=(0.01, 0.2), lambda3_range=(0.1, 0.5),
                 phase_boundaries=(0.3, 0.7), schedule_type='cosine'):
        self.total_epochs = total_epochs
        self.lambda1_range = lambda1_range
        self.lambda2_range = lambda2_range
        self.lambda3_range = lambda3_range
        self.p1, self.p2 = phase_boundaries
        self.schedule_type = schedule_type

    def _interpolate(self, ratio, start, end):
        if self.schedule_type == 'cosine':
            w = 0.5 * (1.0 + _math.cos(_math.pi * ratio))
            return end + (start - end) * w
        return start + (end - start) * ratio

    def get_weights(self, epoch):
        progress = epoch / max(self.total_epochs - 1, 1)
        if progress <= self.p1:
            l1, l2, l3 = self.lambda1_range[0], self.lambda2_range[0], self.lambda3_range[0]
        elif progress <= self.p2:
            ratio = (progress - self.p1) / (self.p2 - self.p1)
            l1 = self._interpolate(ratio, *self.lambda1_range)
            l2 = self._interpolate(ratio, *self.lambda2_range)
            l3 = self._interpolate(ratio, *self.lambda3_range)
        else:
            l1, l2, l3 = self.lambda1_range[1], self.lambda2_range[1], self.lambda3_range[1]
        return {'lambda_phys': l1, 'lambda_orth': l2, 'lambda_smooth': l3}


# ============================================================
# 6. Total Loss Aggregator
# ============================================================
class PhyCDNetTotalLoss(nn.Module):
    """Aggregates all PhyCDNet loss terms with dynamic scheduling."""
    def __init__(self, total_epochs=200, focal_alpha=None,
                 dice_alpha=0.7, dice_beta=0.3, T_ref=0.75, ignore_index=255):
        super().__init__()
        self.cd_loss = ChangeDetectionLossPhy(focal_alpha=focal_alpha,
                                               dice_alpha=dice_alpha, dice_beta=dice_beta,
                                               ignore_index=ignore_index)
        self.phys_loss = PhysicalConsistencyLossPhy(T_ref=T_ref)
        self.orth_loss = OrthogonalDisentanglementLoss()
        self.smooth_loss = TerrainSmoothnessLoss()
        self.scheduler = DynamicWeightScheduler(total_epochs=total_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target, F_path_hat=None, F_A=None, T_eff_hat=None,
                F_terrain=None, F_change=None, phi=None, dem_gradient=None):
        weights = self.scheduler.get_weights(self.current_epoch)
        loss_cd = self.cd_loss(pred, target)
        lp = self.phys_loss(F_path_hat, F_A, T_eff_hat) if F_path_hat is not None else torch.tensor(0.0)
        lo = torch.tensor(0.0)
        if F_terrain is not None and F_change is not None:
            lo = self.orth_loss(F_terrain, F_change)['orth_loss']
        ls = self.smooth_loss(phi, dem_gradient)
        total = loss_cd + weights['lambda_phys'] * lp + \
                weights['lambda_orth'] * lo + weights['lambda_smooth'] * ls
        return {'total': total, 'cd_loss': loss_cd, 'phys_loss': lp,
                'orth_loss': lo, 'smooth_loss': ls, 'weights': weights}
