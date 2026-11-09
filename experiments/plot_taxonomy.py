"""Generate the paper figure: rho vs induction score scatter.

Reads a pilot results JSON (results/invariant_taxonomy_pilot_<model>.json) and
reproduces the original clean scatter: linear axes, top-10% induction heads
highlighted, Spearman r in the title.

Usage:
    uv run experiments/plot_taxonomy.py results/invariant_taxonomy_pilot_smollm2_1_7b.json
    # or a model safe-name; it will look in results/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "results" / "figures"


def load(arg: str) -> dict:
    p = Path(arg)
    if not p.exists():
        p = RESULTS_DIR / arg
    if not p.exists():
        p = RESULTS_DIR / f"invariant_taxonomy_pilot_{arg}.json"
    return json.load(open(p))


def rho_vs_induction(data, tag):
    heads = data["heads"]
    rhos = np.array([h["rho"] for h in heads])
    scores = np.array([h["induction"] for h in heads])
    r = data["spearman_rho_induction"]["r"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(rhos, scores, s=12, alpha=0.5, c="steelblue", edgecolors="none")
    top_mask = scores >= np.percentile(scores, 90)
    ax.scatter(rhos[top_mask], scores[top_mask], s=20, alpha=0.8, c="firebrick",
               edgecolors="none", label="Top 10% induction")

    ax.set_xlabel(r"$\rho$ (positional capacity)", fontsize=11)
    ax.set_ylabel("Induction score", fontsize=11)
    ax.set_title(f"Spearman $r = {r:.2f}$", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"rho_vs_induction_{tag}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("usage: uv run experiments/plot_taxonomy.py <results.json | model_safe_name>")
        raise SystemExit(1)
    data = load(sys.argv[1])
    tag = data["model"].split("/")[-1].lower().replace("-", "_").replace(".", "_")
    print(f"Plotting rho-vs-induction for {data['model']} -> {FIG_DIR}")
    rho_vs_induction(data, tag)
    print("Done.")


if __name__ == "__main__":
    main()
