"""
Physics-motivated evaluation metrics.
  TDI — Terrain Disentanglement Index
  RCQ — Radiometric Compensation Quality 
  MEE — Mean Endpoint Error 
"""

import torch
from typing import Dict


def compute_tdi(f_terrain: torch.Tensor, f_change: torch.Tensor) -> torch.Tensor:
    """TDI = <F_terrain, F_change>_F / (N * C). Lower = better decoupling."""
    B, C, H, W = f_terrain.shape
    inner = (f_terrain * f_change).sum(dim=[1, 2, 3])
    return (inner / (H * W * C)).mean()


def compute_mee(phi: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """MEE = mean(||phi - phi_gt||) in normalized coordinates."""
    B, _, H_f, W_f = phi.shape
    device = phi.device
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H_f, device=device),
                            torch.linspace(-1, 1, W_f, device=device), indexing='ij')
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    ones = torch.ones(B, H_f, W_f, 1, device=device)
    grid_h = torch.cat([grid, ones], dim=-1)
    warped = torch.bmm(affine, grid_h.view(B, -1, 3).transpose(1, 2)).transpose(1, 2)
    warped = warped.view(B, H_f, W_f, 2)
    phi_gt = warped - grid
    phi_pred = phi.permute(0, 2, 3, 1)
    return torch.sqrt(((phi_pred - phi_gt) ** 2).sum(dim=-1)).mean()


def compute_rcq(f_a_norm: torch.Tensor, f_b_r: torch.Tensor) -> torch.Tensor:
    """RCQ = ||F_A_norm - F_B_r||_1 / (N * C). Lower = better radiometric alignment."""
    B, C, H, W = f_a_norm.shape
    l1 = torch.abs(f_a_norm - f_b_r).sum(dim=[1, 2, 3])
    return (l1 / (H * W * C)).mean()


class PhysicsMetricEvaluator:
    """Dataset-level accumulator for TDI, RCQ, MEE."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._tdi_sum = 0.0
        self._rcq_sum = 0.0
        self._mee_sum = 0.0
        self._n_samples = 0

    @torch.no_grad()
    def update(self, f_terrain, f_change, f_a_norm, f_b_r, phi=None, affine=None):
        B = f_terrain.shape[0]
        self._tdi_sum += compute_tdi(f_terrain, f_change).item() * B
        self._rcq_sum += compute_rcq(f_a_norm, f_b_r).item() * B
        if phi is not None and affine is not None:
            try:
                self._mee_sum += compute_mee(phi, affine).item() * B
            except Exception:
                pass
        self._n_samples += B

    def compute(self) -> Dict[str, float]:
        if self._n_samples == 0:
            return {"TDI": float("nan"), "RCQ": float("nan"), "MEE": float("nan")}
        result = {"TDI": self._tdi_sum / self._n_samples,
                   "RCQ": self._rcq_sum / self._n_samples}
        if self._mee_sum > 0:
            result["MEE"] = self._mee_sum / self._n_samples
        return result

    def summary(self) -> str:
        r = self.compute()
        lines = [f"Physics Metrics ({self._n_samples} samples):",
                 f"  TDI = {r['TDI']:.6f}  (lower is better)",
                 f"  RCQ = {r['RCQ']:.6f}  (lower is better)"]
        if 'MEE' in r:
            lines.append(f"  MEE = {r['MEE']:.5f}  (lower is better)")
        return '\n'.join(lines)
