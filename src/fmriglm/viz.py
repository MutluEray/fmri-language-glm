"""Plot helpers (headless-safe)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_design(X, path, title=None):
    Z = X.to_numpy(float)
    Z = Z / np.maximum(np.abs(Z).max(0), 1e-12)
    fig, ax = plt.subplots(figsize=(max(5, 0.4 * Z.shape[1] + 3), 6))
    ax.imshow(Z, aspect="auto", cmap="gray", interpolation="nearest")
    ax.set_xticks(range(Z.shape[1]))
    ax.set_xticklabels(list(X.columns), rotation=90, fontsize=7)
    ax.set_ylabel("scan")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_stat_map(img, path, threshold=None, title=None):
    from nilearn import plotting
    plotting.plot_stat_map(img, threshold=threshold, title=title, output_file=str(path))
    plt.close("all")
