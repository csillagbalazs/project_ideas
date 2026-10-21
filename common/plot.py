"""Matplotlib style for the one-page summaries (reference palette, light mode)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 8.5,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=110)
    plt.close(fig)
