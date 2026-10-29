"""Command line entry point: ``irisk <stage>``.

    irisk data       # download NHANES cycles
    irisk cohort     # build the analysis cohort and print the participant flow
    irisk evaluate   # score the model against the clinical indices
    irisk train      # fit on everything and export a serving artefact
    irisk report     # collect metrics and figures into reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from irisk import plots
from irisk.cohort import build_cohort, weighted_prevalence, weighted_quantile
from irisk.evaluate import calibration, net_benefit, score, treat_all_net_benefit, weighted_auroc
from irisk.indices import HOMA_IR_THRESHOLDS
from irisk.model import (
    by_cycle_split,
    clinical_index_scores,
    cross_validated_scores,
    fit,
)
from irisk.nhanes import CYCLES, DEFAULT_CYCLES, download_cycle, load_cycles

DATA = Path("data/nhanes")
REPORTS = Path("reports")
MODELS = Path("models")


def _cohort(args: argparse.Namespace):
    frame = load_cycles(tuple(args.cycles), DATA)
    return build_cohort(frame)


def cmd_data(args: argparse.Namespace) -> None:
    for name in args.cycles:
        paths = download_cycle(CYCLES[name], DATA)
        total = sum(p.stat().st_size for p in paths.values()) / 1e6
        print(f"{name}  {len(paths)} components  {total:.0f} MB")


def cmd_cohort(args: argparse.Namespace) -> None:
    cohort, report = _cohort(args)
    print("Participant flow:")
    print(report.describe())

    weights = cohort["weight"].to_numpy()
    homa = cohort["homa_ir"].to_numpy()
    print(f"\nHOMA-IR, survey weighted: median {weighted_quantile(homa, 0.5, weights):.2f} "
          f"(IQR {weighted_quantile(homa, 0.25, weights):.2f}"
          f" to {weighted_quantile(homa, 0.75, weights):.2f})")
    for threshold in HOMA_IR_THRESHOLDS:
        labels = homa >= threshold.value
        print(f"  insulin resistant at {threshold.value:.2f}: "
              f"{weighted_prevalence(labels, weights) * 100:.1f}% weighted, "
              f"{labels.mean() * 100:.1f}% unweighted   [{threshold.population}]")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cohort, report = _cohort(args)
    cohort = cohort.reset_index(drop=True)
    weights = cohort["weight"].to_numpy()
    labels = (cohort["homa_ir"] >= args.threshold).astype(int).to_numpy()

    out_of_fold = cross_validated_scores(cohort, args.threshold, n_splits=args.folds)
    cards = [score("Gradient boosting (no insulin)", labels, out_of_fold, weights,
                   probabilities=out_of_fold)]
    indices = clinical_index_scores(cohort)
    for name, values in indices.items():
        cards.append(score(name, labels, np.asarray(values, dtype=float), weights))

    # Does it survive a survey wave it never saw?
    temporal: dict[str, Any] = {}
    for cycle in sorted(cohort["cycle"].unique()):
        train, test = by_cycle_split(cohort, cycle)
        model = fit(train, args.threshold)
        test_labels = (test["homa_ir"] >= args.threshold).astype(int).to_numpy()
        test_weights = test["weight"].to_numpy()
        test_indices = clinical_index_scores(test)
        temporal[cycle] = {
            "n": int(len(test)),
            "model": weighted_auroc(test_labels, model.predict_proba(test), test_weights),
            "waist": weighted_auroc(test_labels, test_indices["Waist alone"], test_weights),
            "tyg": weighted_auroc(test_labels, test_indices["TyG index"], test_weights),
        }

    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, Any] = {}
    n = len(labels)
    for name in ("Waist alone", "TyG index", "BMI alone"):
        reference = np.asarray(indices[name], dtype=float)
        differences = []
        for _ in range(args.bootstrap):
            draw = rng.integers(0, n, n)
            if labels[draw].sum() in (0, len(draw)):
                continue
            differences.append(
                weighted_auroc(labels[draw], out_of_fold[draw], weights[draw])
                - weighted_auroc(labels[draw], reference[draw], weights[draw])
            )
        differences = np.array(differences)
        comparisons[f"model vs {name}"] = {
            "mean_difference": float(differences.mean()),
            "ci_low": float(np.percentile(differences, 2.5)),
            "ci_high": float(np.percentile(differences, 97.5)),
            "p_better": float(np.mean(differences > 0)),
        }

    curve = {
        str(t): {
            "model": net_benefit(labels, out_of_fold, weights, t),
            "treat_all": treat_all_net_benefit(labels, weights, t),
        }
        for t in (0.2, 0.3, 0.4, 0.5, 0.6)
    }

    record = {
        "threshold": args.threshold,
        "cycles": list(args.cycles),
        "cohort_flow": report.steps,
        "n": int(len(cohort)),
        "weighted_prevalence": weighted_prevalence(labels, weights),
        "scorecards": [vars(c) for c in cards],
        "temporal_validation": temporal,
        "comparisons": comparisons,
        "calibration": calibration(labels, out_of_fold, weights),
        "decision_curve": curve,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps(record, indent=2))

    print(f"n = {len(cohort)}, weighted prevalence "
          f"{record['weighted_prevalence'] * 100:.1f}% at HOMA-IR >= {args.threshold}\n")
    print(f"{'model':32s} {'AUROC':>6s} {'Sens@90Spec':>12s} {'Spec@90Sens':>12s} {'Brier':>7s}")
    for card in sorted(cards, key=lambda c: -c.auroc):
        print(f"{card.name:32s} {card.auroc:6.3f} {card.sensitivity_at_90_spec:12.3f} "
              f"{card.specificity_at_90_sens:12.3f} {card.brier:7.4f}")
    print()
    for label, test in comparisons.items():
        print(f"{label:28s} {test['mean_difference']:+.3f} "
              f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}]  P={test['p_better']:.3f}")


def cmd_train(args: argparse.Namespace) -> None:
    cohort, _ = _cohort(args)
    model = fit(cohort, args.threshold)
    path = model.save(MODELS / "model.joblib")
    print(f"trained on {model.trained_on['n']} participants from "
          f"{', '.join(model.trained_on['cycles'])} -> {path}")
    print(f"boosting iterations: {model.trained_on['n_iterations']}")


def cmd_report(args: argparse.Namespace) -> None:
    path = REPORTS / "evaluation.json"
    if not path.exists():
        raise SystemExit("missing reports/evaluation.json; run `irisk evaluate` first")
    record = json.loads(path.read_text())

    cohort, _ = _cohort(args)
    plots.score_comparison(record, REPORTS / "auroc.png")
    plots.calibration_curve(record, REPORTS / "calibration.png")
    plots.decision_curve(record, REPORTS / "decision_curve.png")
    plots.homa_distribution(cohort, REPORTS / "homa_distribution.png")

    lines = [
        "# Results",
        "",
        f"NHANES {', '.join(record['cycles'])}, {record['n']} adults. "
        f"Insulin resistance defined as HOMA-IR >= {record['threshold']}, "
        f"survey-weighted prevalence {record['weighted_prevalence'] * 100:.1f}%.",
        "",
        "| Model | AUROC | Sens @ 90% spec | Spec @ 90% sens | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for card in sorted(record["scorecards"], key=lambda c: -c["auroc"]):
        brier = "n/a" if np.isnan(card["brier"]) else f"{card['brier']:.4f}"
        lines.append(
            f"| {card['name']} | {card['auroc']:.3f} | "
            f"{card['sensitivity_at_90_spec']:.3f} | "
            f"{card['specificity_at_90_sens']:.3f} | {brier} |"
        )

    lines += ["", "## Held-out survey wave", "",
              "| Tested on | n | Model | Waist | TyG |", "|---|---:|---:|---:|---:|"]
    for cycle, values in record["temporal_validation"].items():
        lines.append(f"| {cycle} | {values['n']} | {values['model']:.3f} | "
                     f"{values['waist']:.3f} | {values['tyg']:.3f} |")

    lines += ["", "## Paired bootstrap", "",
              "| Comparison | Delta AUROC | 95% CI | P(better) |", "|---|---:|---|---:|"]
    for label, test in record["comparisons"].items():
        lines.append(f"| {label} | {test['mean_difference']:+.3f} | "
                     f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}] | "
                     f"{test['p_better']:.3f} |")

    lines += ["", "![AUROC](auroc.png)", "", "![calibration](calibration.png)", ""]
    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="irisk", description=__doc__)
    parser.add_argument("--cycles", nargs="+", default=list(DEFAULT_CYCLES))
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="HOMA-IR cut-off defining insulin resistance")
    parser.add_argument("--seed", type=int, default=7)
    sub = parser.add_subparsers(dest="stage", required=True)

    for name, handler in (("data", cmd_data), ("cohort", cmd_cohort),
                          ("train", cmd_train), ("report", cmd_report)):
        sub.add_parser(name, help=name).set_defaults(handler=handler)

    evaluate_parser = sub.add_parser("evaluate", help="score against the clinical indices")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.set_defaults(handler=cmd_evaluate)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
