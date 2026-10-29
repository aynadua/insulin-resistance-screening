"""Figures for the README. Headless by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def score_comparison(record: dict[str, Any], path: str | Path) -> Path:
    """AUROC and sensitivity at 90% specificity, model against each index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cards = sorted(record["scorecards"], key=lambda c: c["auroc"])
    names = [c["name"] for c in cards]
    auroc = [c["auroc"] for c in cards]
    sensitivity = [c["sensitivity_at_90_spec"] for c in cards]
    colours = ["tab:blue" if "boosting" in n else "tab:grey" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    axes[0].barh(names, auroc, color=colours)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].axvline(0.5, color="black", linewidth=1)
    axes[0].set_xlabel("AUROC (survey weighted)")
    axes[0].set_title("Discrimination", fontsize=10)

    axes[1].barh(names, sensitivity, color=colours)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("sensitivity at 90% specificity")
    axes[1].set_title("Usable operating point", fontsize=10)

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Predicting insulin resistance without an insulin assay", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def calibration_curve(record: dict[str, Any], path: str | Path) -> Path:
    """Predicted against observed risk, by weighted decile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calibrated = record["calibration"]

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    ax.plot(calibrated["predicted"], calibrated["observed"], marker="o",
            markersize=5, label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed proportion")
    ax.set_title(f"Calibration (Brier {calibrated['brier']:.3f})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def decision_curve(record: dict[str, Any], path: str | Path) -> Path:
    """Net benefit against threshold probability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = record["decision_curve"]
    thresholds = sorted(float(t) for t in curve)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
    ax.plot(thresholds, [curve[str(t)]["model"] for t in thresholds],
            marker="o", markersize=4, label="model")
    ax.plot(thresholds, [curve[str(t)]["treat_all"] for t in thresholds],
            linestyle="--", label="treat everyone")
    ax.axhline(0.0, color="black", linewidth=1, label="treat nobody")

    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def homa_distribution(cohort: pd.DataFrame, path: str | Path) -> Path:
    """Weighted HOMA-IR distribution, with the published thresholds marked."""
    from irisk.indices import HOMA_IR_THRESHOLDS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    homa = cohort["homa_ir"].to_numpy()
    weights = cohort["weight"].to_numpy()
    keep = homa < 20  # the tail runs past 100 and would flatten the plot

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.hist(homa[keep], bins=80, weights=weights[keep], color="tab:blue", alpha=0.75)
    for threshold in HOMA_IR_THRESHOLDS:
        ax.axvline(threshold.value, linestyle="--", linewidth=1.2,
                   label=f"{threshold.value} ({threshold.population[:28]})")

    ax.set_xlabel("HOMA-IR")
    ax.set_ylabel("weighted count")
    ax.set_title("HOMA-IR has no natural cut-point, which is why thresholds disagree",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
