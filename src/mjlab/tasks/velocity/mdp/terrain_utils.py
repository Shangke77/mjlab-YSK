"""Terrain normal estimation from raycast hit points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def fit_terrain_normal(
  points: torch.Tensor,
  valid_mask: torch.Tensor,
) -> torch.Tensor:
  """Fit a plane normal from 3D points via covariance eigendecomposition.

  Args:
    points: [B, N, 3] world-frame positions.
    valid_mask: [B, N] boolean (True = valid).

  Returns:
    [B, 3] unit normal oriented upward. Falls back to [0, 0, 1]
    when fewer than 3 finite valid points or when the fit is degenerate.
  """
  B = points.shape[0]
  device = points.device

  # Invalid ray hits may contain NaN or Inf coordinates. Multiplying those
  # coordinates by a zero mask would still produce NaN, so exclude them and
  # replace them explicitly before accumulating the covariance.
  valid_mask = valid_mask & torch.isfinite(points).all(dim=-1)
  count = valid_mask.sum(dim=1)
  enough = count >= 3

  mask = valid_mask.unsqueeze(-1)
  masked_points = torch.where(mask, points, torch.zeros_like(points))
  count_clamped = count.clamp(min=1).float().unsqueeze(-1)
  centroid = masked_points.sum(dim=1) / count_clamped
  centered = torch.where(
    mask,
    points - centroid.unsqueeze(1),
    torch.zeros_like(points),
  )

  cov = torch.einsum("bni,bnj->bij", centered, centered)
  cov = 0.5 * (cov + cov.transpose(-1, -2))

  # Normalize each covariance to keep the eigensolver in a stable numeric
  # range. Unusable batches get a finite matrix with distinct eigenvalues so
  # one bad ray cloud cannot make the batched GPU eigendecomposition fail.
  scale = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
  usable = enough & torch.isfinite(cov).all(dim=(-2, -1)) & (scale > 0)
  tiny = torch.finfo(cov.dtype).tiny
  normalized_cov = cov / scale.clamp(min=tiny).view(B, 1, 1)
  fallback_cov = torch.diag(
    torch.tensor([1.0, 2.0, 3.0], device=device, dtype=cov.dtype)
  ).expand(B, -1, -1)
  normalized_cov = torch.where(usable.view(B, 1, 1), normalized_cov, fallback_cov)

  # A tiny unequal diagonal perturbation separates repeated eigenvalues in
  # line-like and point-like clouds. It is negligible for a valid plane and
  # those degenerate fits are rejected below.
  regularizer = torch.diag(
    torch.tensor([1.0e-6, 2.0e-6, 3.0e-6], device=device, dtype=cov.dtype)
  )
  eigenvalues, eigenvectors = torch.linalg.eigh(normalized_cov + regularizer)
  normal = eigenvectors[:, :, 0]  # Smallest eigenvalue = plane normal.
  normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

  # Orient upward.
  normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)

  # Fall back when the fit is degenerate (collinear or near-duplicate
  # points). A valid plane has one small eigenvalue and two materially
  # larger ones; line-like or point-like clouds do not.
  eps = torch.finfo(eigenvalues.dtype).eps
  plane_like = (eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=eps)) < 0.1
  has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=eps) * 1e-6
  reliable = usable & plane_like & has_spread

  up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
  return torch.where(reliable.unsqueeze(-1), normal, up)


# Cached subsample indices to avoid per-step allocation.
_subsample_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _subsample_indices(
  total: int, max_points: int, device: torch.device
) -> torch.Tensor:
  key = (total, max_points, device)
  if key not in _subsample_cache:
    _subsample_cache[key] = torch.linspace(
      0, total - 1, max_points, device=device
    ).long()
  return _subsample_cache[key]


def terrain_normal_from_sensors(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  max_points: int = 32,
) -> torch.Tensor:
  """Estimate terrain normal from one or more raycast sensors.

  Gathers hit positions, subsamples to *max_points* per sensor, and fits a plane
  normal via :func:`fit_terrain_normal`.

  Returns:
    [B, 3] unit terrain normal in world frame.
  """
  all_points: list[torch.Tensor] = []
  all_valid: list[torch.Tensor] = []

  for name in sensor_names:
    sensor = env.scene[name]
    if not isinstance(sensor, RayCastSensor):
      raise TypeError(
        f"Sensor '{name}' is {type(sensor).__name__}, expected RayCastSensor."
      )

    hit_pos = sensor.data.hit_pos_w
    valid = sensor.data.distances >= 0

    N = hit_pos.shape[1]
    if N > max_points:
      idx = _subsample_indices(N, max_points, hit_pos.device)
      hit_pos = hit_pos[:, idx]
      valid = valid[:, idx]

    all_points.append(hit_pos)
    all_valid.append(valid)

  points = torch.cat(all_points, dim=1)
  valid_mask = torch.cat(all_valid, dim=1)
  return fit_terrain_normal(points, valid_mask)
