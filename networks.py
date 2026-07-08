"""
PhyCDNet — Physics-guided Causal Heterogeneous Change Detection Network.
Three core modules: CRMA (Sec. 3.2), PCOD (Sec. 3.3), CFDA (Sec. 3.4).

Architecture overview:
  Input A (Pre-disaster) --→ [CRMA] --→ [PCOD] ----------→ [Decoder] --→ Change Map
  Input B (Post-disaster) ---------------------------------↗      ↑
                            Input A --→ [CFDA] ←-- Input B --------↗

Usage:
  from models.networks import PhyCDNet
  model = PhyCDNet(backbone='resnet18', feature_dim=256, n_class=2)
  output = model(image_A, image_B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# 1. Degradation Simulator (Sec. 3.1: Cross-sensor preconditioning)
# ============================================================
class DegradationSimulator(nn.Module):
    """Simulates geometric misalignment + resolution downgrade on pre-event image."""
    def __init__(self, max_translation=15, max_rotation=5.0, downsample_factor=4.0, img_size=256):
        super().__init__()
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.downsample_factor = downsample_factor
        self.img_size = img_size

    def _random_affine(self, B, H, W, device):
        angle_deg = (torch.rand(B, device=device) * 2 - 1) * self.max_rotation
        theta = angle_deg * (math.pi / 180.0)
        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        tx = (torch.rand(B, device=device) * 2 - 1) * self.max_translation / (W / 2.0)
        ty = (torch.rand(B, device=device) * 2 - 1) * self.max_translation / (H / 2.0)
        affine = torch.zeros(B, 2, 3, device=device)
        affine[:, 0, 0] = cos_a;  affine[:, 0, 1] = -sin_a; affine[:, 0, 2] = tx
        affine[:, 1, 0] = sin_a;  affine[:, 1, 1] =  cos_a; affine[:, 1, 2] = ty
        return affine

    def forward(self, x, label=None):
        B, C, H, W = x.shape
        affine = self._random_affine(B, H, W, x.device)
        grid = F.affine_grid(affine, x.size(), align_corners=False)
        x_warped = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        x_down = F.interpolate(x_warped, scale_factor=1.0/self.downsample_factor,
                               mode='bilinear', align_corners=False)
        x_simulated = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)
        label_out = None
        if label is not None:
            label_out = F.grid_sample(label.float(), grid, mode='nearest', align_corners=False)
        return x_simulated, affine, label_out


# ============================================================
# 2. Dual Encoder (shared ResNet-18 backbone)
# ============================================================
class DualResNetEncoder(nn.Module):
    """Two independent ResNet-18 backbones for pre-event (LR) and post-event (HR)."""
    def __init__(self, pretrained=True, output_channels=256):
        super().__init__()
        import torchvision.models as tv_models
        for prefix in ['a', 'b']:
            resnet = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            setattr(self, f'conv1_{prefix}', resnet.conv1)
            setattr(self, f'bn1_{prefix}', resnet.bn1)
            setattr(self, f'relu_{prefix}', resnet.relu)
            setattr(self, f'maxpool_{prefix}', resnet.maxpool)
            for l in [1, 2, 3, 4]:
                setattr(self, f'layer{l}_{prefix}', getattr(resnet, f'layer{l}'))
        self.compress_a = nn.Conv2d(512, output_channels, kernel_size=1)
        self.compress_b = nn.Conv2d(512, output_channels, kernel_size=1)

    def _forward_one(self, x, prefix):
        x = getattr(self, f'conv1_{prefix}')(x)
        x = getattr(self, f'bn1_{prefix}')(x)
        x = getattr(self, f'relu_{prefix}')(x)
        x = getattr(self, f'maxpool_{prefix}')(x)
        for l in [1, 2, 3]:
            x = getattr(self, f'layer{l}_{prefix}')(x)
        return getattr(self, f'layer4_{prefix}')(x)

    def forward(self, I_A, I_B):
        F_A = self.compress_a(self._forward_one(I_A, 'a'))
        F_B = self.compress_b(self._forward_one(I_B, 'b'))
        return F_A, F_B


# ============================================================
# 3. CRMA Module (Sec. 3.2)
#    PDFN: Physics-grounded Degenerate Feature Normalization
#    RMDC: Resolution-aware Multi-scale Dilated Compensation
# ============================================================
class CRMAModule(nn.Module):
    """Cross-Resolution Physical Manifold Alignment.
    Implements the RTE inverse: F_A^rad = (F_A - F_path) / |T_eff|
    """
    def __init__(self, channels=256, reduction=8, dilate_rate=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Path radiance estimation: F_path(w_p)
        self.path_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1))
        # Learnable transmittance vector: T_eff(w_t) ∈ R^C
        self.T_eff = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.8)
        # RMDC: dilated convolution for resolution alignment
        self.dil_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                  padding=dilate_rate, dilation=dilate_rate, bias=False)
        self.dil_norm = nn.InstanceNorm2d(channels)
        self.dil_act = nn.ReLU(inplace=True)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels))

    def forward(self, F_A):
        F_path = self.path_mlp(self.avg_pool(F_A))
        F_phys = (F_A - F_path) / (torch.abs(self.T_eff) + 1e-6)
        F_dil = self.dil_act(self.dil_norm(self.dil_conv(F_phys)))
        F_A_rad = F_phys + self.refine(F_dil)
        return F_A_rad, F_path, self.T_eff


# ============================================================
# 4. PCOD Module (Sec. 3.3)
#    Physics-Causal Object Decoupling with orthogonality constraint
# ============================================================
class PCODModule(nn.Module):
    """Causal terrain-change decoupling via channel-wise gating.
    F_change = F_A_rad - F_terrain, with F_terrain = w ⊙ F_A_rad.
    Enforces orthogonality: R_orth = ||F_terrain^T @ F_change||^2_F
    """
    def __init__(self, channels=256, terrain_dim=3, d=32):
        super().__init__()
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(terrain_dim, 16, 3, padding=1), nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, d, 3, padding=1), nn.AdaptiveAvgPool2d(1))
        self.projection_net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(inplace=True),
            nn.Linear(d * 2, channels))

    def forward(self, F_A_rad, terrain_map=None):
        B, C, H, W = F_A_rad.shape
        if terrain_map is None:
            terrain_emb = torch.zeros(B, self.terrain_encoder[-1].out_channels,
                                       device=F_A_rad.device)
        else:
            if terrain_map.shape[-2:] != (H, W):
                terrain_map = F.interpolate(terrain_map, size=(H, W),
                                            mode='bilinear', align_corners=False)
            terrain_emb = self.terrain_encoder(terrain_map).squeeze(-1).squeeze(-1)
        proj_weights = torch.tanh(self.projection_net(terrain_emb).view(B, C, 1, 1))
        F_terrain = proj_weights * F_A_rad
        F_change = F_A_rad - F_terrain
        # Orthogonality constraint
        F_t_flat = F_terrain.view(B, C, -1)
        F_c_flat = F_change.view(B, C, -1)
        cross_corr = torch.bmm(F_t_flat, F_c_flat.transpose(1, 2))
        orth_loss = (cross_corr ** 2).mean()
        return F_change, F_terrain, orth_loss


# ============================================================
# 5. CFDA Module (Sec. 3.4)
#    Continuous Feature-flow Dynamic Alignment
#    with terrain-adaptive smoothness regularization
# ============================================================
class CorrelationEstimator(nn.Module):
    """Multi-scale feature correlation → dense displacement field φ."""
    def __init__(self, channels=256):
        super().__init__()
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True))
        self.refine_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.refine_conv2 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels // 4), nn.ReLU(inplace=True))
        self.flow_head = nn.Sequential(
            nn.Conv2d(channels // 4, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1), nn.Tanh())

    def forward(self, F_change, F_B):
        H, W = F_change.shape[-2:]
        F_B_r = F.interpolate(F_B, size=(H, W), mode='bilinear', align_corners=False)
        concat = torch.cat([F_change, F_B_r], dim=1)
        coarse = self.coarse_conv(concat)
        r1 = self.refine_conv1(coarse)
        r2 = self.refine_conv2(r1)
        return self.flow_head(r2)


class CFDAModule(nn.Module):
    """Continuous Feature-flow Dynamic Alignment.
    Warp F_B via φ: F_B^aligned = Warp(F_B, φ)
    Terrain-adaptive smoothness: R_terrain = Σ ||∇φ||² / (||∇h|| + α)
    """
    def __init__(self, channels=256, alpha=1e-3):
        super().__init__()
        self.alpha = alpha
        self.corr_estimator = CorrelationEstimator(channels=channels)

    def _warp(self, F_B, phi):
        B, C, h_out, w_out = phi.shape[0], F_B.shape[1], phi.shape[2], phi.shape[3]
        theta = torch.eye(2, 3, device=phi.device).unsqueeze(0).repeat(B, 1, 1)
        base_grid = F.affine_grid(theta, (B, 1, h_out, w_out), align_corners=False)
        sample_grid = base_grid + phi.permute(0, 2, 3, 1)
        return F.grid_sample(F_B, sample_grid, mode='bilinear',
                              padding_mode='border', align_corners=False)

    def _smooth_loss(self, phi, dem_gradient=None):
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

    def forward(self, F_change, F_B, dem_gradient=None):
        phi = self.corr_estimator(F_change, F_B)
        F_B_aligned = self._warp(F_B, phi)
        smooth_loss = self._smooth_loss(phi, dem_gradient)
        return F_B_aligned, phi, smooth_loss


# ============================================================
# 6. Difference Decoder
# ============================================================
class DifferenceDecoder(nn.Module):
    """|F_change - F_B_aligned| → change logits."""
    def __init__(self, in_channels=256, mid_channels=128, out_channels=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1))

    def forward(self, F_change, F_B_aligned, target_size=None):
        out = self.decoder(torch.abs(F_change - F_B_aligned))
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out


# ============================================================
# 7. PhyCDNet — Full Model
# ============================================================
class PhyCDNet(nn.Module):
    """Physics-guided Causal Heterogeneous Change Detection Network.
    End-to-end training with three complementary losses:
      L_total = L_CD + λ_phys·L_phys + λ_orth·R_orth + λ_smooth·R_terrain
    """
    def __init__(self, backbone='resnet18', feature_dim=256, n_class=2,
                 pretrained=True, use_dem=False):
        super().__init__()
        self.encoder = DualResNetEncoder(pretrained=pretrained, output_channels=feature_dim)
        self.crma = CRMAModule(channels=feature_dim)
        self.pcod = PCODModule(channels=feature_dim, terrain_dim=3, d=32)
        self.cfda = CFDAModule(channels=feature_dim)
        self.decoder = DifferenceDecoder(in_channels=feature_dim, mid_channels=128,
                                          out_channels=n_class)
        self.simulator = DegradationSimulator()

    def forward(self, I_A, I_B, dem_data=None, label=None):
        I_A_original = I_A.clone()
        I_A, affine_mat, label_warped = self.simulator(I_A, label=label)
        F_A, F_B = self.encoder(I_A, I_B)

        # CRMA: radiometric correction
        F_A_rad, F_path, T_eff = self.crma(F_A)

        # PCOD: causal terrain-change decoupling
        terrain_map = dem_data.get('terrain', None) if isinstance(dem_data, dict) else None
        F_change, F_terrain, orth_loss = self.pcod(F_A_rad, terrain_map)

        # CFDA: cross-sensor feature alignment
        dem_gradient = dem_data.get('gradient', None) if isinstance(dem_data, dict) else None
        F_B_aligned, phi, smooth_loss = self.cfda(F_change, F_B, dem_gradient)

        # Decoder
        C_hat = self.decoder(F_change, F_B_aligned,
                             target_size=(I_B.shape[-2], I_B.shape[-1]))

        # Physical consistency loss
        phys_loss = F.mse_loss(F_path, F_A.detach()) + \
                    F.mse_loss(torch.abs(T_eff), torch.ones_like(T_eff) * 0.8)

        return {
            'pred': C_hat,
            'phys_loss': phys_loss, 'orth_loss': orth_loss, 'smooth_loss': smooth_loss,
            'F_A': F_A, 'F_B': F_B, 'F_B_aligned': F_B_aligned,
            'F_change': F_change, 'F_terrain': F_terrain, 'F_A_rad': F_A_rad,
            'phi': phi, 'affine_mat': affine_mat,
            'I_A_original': I_A_original, 'I_A_simulated': I_A,
        }
