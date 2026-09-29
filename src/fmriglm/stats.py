"""p-values and effect combination."""
import numpy as np
from scipy import stats


def t_to_p(t, dof, alternative="greater"):
    if alternative == "greater":
        return stats.t.sf(t, dof)
    if alternative == "two-sided":
        return 2 * stats.t.sf(np.abs(t), dof)
    raise ValueError(alternative)


def p_to_z(p):
    return stats.norm.isf(np.clip(p, 1e-300, 1.0))


def combine_fixed(effects, variances):
    """Fixed-effects average across runs of one subject."""
    n = len(effects)
    return np.mean(effects, 0), np.sum(variances, 0) / n ** 2


def one_sample_t(effects):
    """Group-level one-sample t across subjects. effects: (S, V)."""
    S = effects.shape[0]
    m = effects.mean(0)
    sd = effects.std(0, ddof=1)
    return m / np.maximum(sd / np.sqrt(S), 1e-12), S - 1
