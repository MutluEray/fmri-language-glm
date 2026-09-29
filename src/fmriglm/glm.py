"""Voxelwise GLM: OLS and AR(1) prewhitening, written from scratch."""
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class GLMResult:
    beta: np.ndarray            # (p, V)
    sigma2: np.ndarray          # (V,)
    dof: float
    groups: List[Any]           # [(voxel_idx, (X'X)^-1 for that group)]
    lag1: np.ndarray            # (V,) lag-1 autocorrelation of final residuals
    rho: Optional[np.ndarray] = None   # AR(1) coefficient per voxel (ar1 only)


def lag1_autocorr(resid):
    num = (resid[1:] * resid[:-1]).sum(0)
    den = np.maximum((resid ** 2).sum(0), 1e-12)
    return num / den


def _ols(X, Y):
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    resid = Y - X @ beta
    return beta, resid, pinv @ pinv.T


def _whiten(A, rho):
    out = A.copy()
    out[1:] = A[1:] - rho * A[:-1]
    out[0] = A[0] * np.sqrt(1.0 - rho ** 2)
    return out


def fit_glm(Y, X, noise_model="ols", rho_step=0.01):
    """Fit Y (T x V) = X (T x p) beta + noise.

    noise_model: 'ols'; 'ar1' (per-voxel AR(1) estimated from OLS residuals, quantised to `rho_step`,
    then voxels sharing a rho are refit after prewhitening); 'ar1_global' (one rho for all voxels: the
    median of the per-voxel estimates); and the bias-corrected variants 'ar1_bc' / 'ar1_global_bc', which
    subtract the exact expected lag-1 bias of residuals from a p-regressor fit to white noise,
    b = tr(R L) / tr(R) with R = I - X X^+ and L the symmetrised lag-1 shift matrix. Without this correction the
    residual-based rho is biased negative (about -p/T) and whitening slightly inflates false positives.
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    T, V = Y.shape
    dof = float(T - np.linalg.matrix_rank(X))
    beta, resid, xtxi = _ols(X, Y)
    if noise_model == "ols":
        sigma2 = (resid ** 2).sum(0) / dof
        return GLMResult(beta, sigma2, dof, [(np.arange(V), xtxi)], lag1_autocorr(resid))
    bc = noise_model.endswith("_bc")
    base = noise_model[:-3] if bc else noise_model
    if base not in ("ar1", "ar1_global"):
        raise ValueError(f"unknown noise_model {noise_model!r}")
    rho = lag1_autocorr(resid)
    if bc:
        Rm = np.eye(T) - X @ np.linalg.pinv(X)
        Lm = 0.5 * (np.eye(T, k=1) + np.eye(T, k=-1))
        rho = rho - np.trace(Rm @ Lm) / np.trace(Rm)
    if base == "ar1_global":
        rho = np.full(V, np.median(rho))
    rho = np.clip(np.round(rho / rho_step) * rho_step, -0.95, 0.95)
    beta = np.empty_like(beta)
    sigma2 = np.empty(V)
    resid_w = np.empty_like(resid)
    groups = []
    for r in np.unique(rho):
        idx = np.flatnonzero(rho == r)
        b, res, xtxi_w = _ols(_whiten(X, r), _whiten(Y[:, idx], r))
        beta[:, idx] = b
        resid_w[:, idx] = res
        sigma2[idx] = (res ** 2).sum(0) / dof
        groups.append((idx, xtxi_w))
    return GLMResult(beta, sigma2, dof, groups, lag1_autocorr(resid_w), rho)


def contrast(result, c):
    """Contrast effect c'beta and its variance, per voxel."""
    c = np.asarray(c, float)
    effect = c @ result.beta
    var = np.empty_like(effect)
    for idx, xtxi in result.groups:
        var[idx] = result.sigma2[idx] * float(c @ xtxi @ c)
    return effect, var


def t_stat(effect, var):
    return effect / np.sqrt(np.maximum(var, 1e-20))
