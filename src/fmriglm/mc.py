"""Multiple-comparison procedures."""
import numpy as np
from scipy import ndimage


def bonferroni(p, alpha=0.05):
    p = np.asarray(p)
    return p <= alpha / p.size


def fdr_bh(p, q=0.05):
    """Benjamini-Hochberg. Returns (boolean mask, p cutoff)."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    ps = p[order]
    below = np.flatnonzero(ps <= q * np.arange(1, m + 1) / m)
    if below.size == 0:
        return np.zeros(m, bool), 0.0
    cutoff = ps[below.max()]
    return p <= cutoff, float(cutoff)


def cluster_extent(vol_bool, min_size, connectivity=1):
    """Keep connected components of a 3D boolean volume with >= min_size voxels.

    NOTE: cluster-forming threshold + fixed extent is a heuristic, not FWE control.
    """
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(vol_bool, structure=structure)
    if n == 0:
        return np.zeros_like(vol_bool, bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def sign_flip_maxT(effects, n_perm=5000, seed=0, chunk=100):
    """One-sample max-T permutation test by random sign flipping (FWE-corrected).

    effects: (S, V) subject-level effect maps. Assumes the null distribution of
    each subject's effect is symmetric about zero. If 2**S <= n_perm all sign
    patterns are enumerated exactly.
    Returns dict(t, p_fwe, max_null, exact).
    """
    effects = np.asarray(effects, float)
    S, V = effects.shape
    if S < 3:
        raise ValueError("need at least 3 subjects")
    ss = (effects ** 2).sum(0)

    def tmap(m):
        var = (ss - S * m ** 2) / (S - 1)
        return m / np.sqrt(np.maximum(var, 1e-12) / S)

    obs_t = tmap(effects.mean(0))
    exact = 2 ** S <= n_perm
    if exact:
        bits = (np.arange(2 ** S)[:, None] >> np.arange(S)) & 1
        signs_all = (2 * bits - 1).astype(float)
        n_total = signs_all.shape[0]
    else:
        rng = np.random.default_rng(seed)
        n_total = n_perm
    max_null = np.empty(n_total)
    done = 0
    while done < n_total:
        k = min(chunk, n_total - done)
        signs = signs_all[done:done + k] if exact else rng.choice([-1.0, 1.0], size=(k, S))
        max_null[done:done + k] = tmap(signs @ effects / S).max(1)
        done += k
    srt = np.sort(max_null)
    count = n_total - np.searchsorted(srt, obs_t, side="left")
    p_fwe = count / n_total if exact else (count + 1) / (n_total + 1)
    return dict(t=obs_t, p_fwe=p_fwe, max_null=max_null, exact=exact)
