"""Paired small-sample statistics for the Holo/Qwen Layer-15 causal panel."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data/remote-results/qwen-holo-action-panel-v1-20260922-phase-b-layer15/run"
    / "qwen-holo-action-panel-v1-20260922-phase-b-layer15"
)
DEFAULT_OUTPUT = ROOT / "artifacts/screenspot-presentation/causal-panel-statistics-v1.json"
INTERVENTION = "layer_15_coordinate_x_residuals"


def _restoration(path: Path) -> float:
    payload = json.loads(path.read_text())
    for intervention in payload["interventions"]:
        if intervention["name"] == INTERVENTION:
            return float(intervention["restoration_nats"])
    raise ValueError(f"{INTERVENTION!r} missing from {path}")


def _paired_values(root: Path) -> list[dict[str, float | str]]:
    holo_root = root / "holo"
    qwen_root = root / "qwen"
    cases = sorted(path.parent.name for path in holo_root.glob("*/results.json"))
    rows: list[dict[str, float | str]] = []
    for case in cases:
        holo = _restoration(holo_root / case / "results.json")
        qwen = _restoration(qwen_root / case / "results.json")
        rows.append({"case": case, "holo_nats": holo, "qwen_nats": qwen, "difference_nats": holo - qwen})
    if len(rows) != 8:
        raise ValueError(f"expected eight paired prompts, found {len(rows)}")
    return rows


def _bootstrap_median_ci(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_p(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    null_means = [
        float(np.mean(values * np.asarray(signs, dtype=np.float64)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-15 for value in null_means) / len(null_means)


def _exact_positive_sign_p(positive: int, count: int) -> float:
    numerator = sum(math.comb(count, k) for k in range(positive, count + 1))
    return numerator / (2**count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20_260_922)
    args = parser.parse_args()

    rows = _paired_values(args.input)
    holo = np.asarray([row["holo_nats"] for row in rows], dtype=np.float64)
    qwen = np.asarray([row["qwen_nats"] for row in rows], dtype=np.float64)
    difference = holo - qwen
    ci_low, ci_high = _bootstrap_median_ci(difference, seed=args.seed, draws=args.bootstrap_draws)
    median_difference = float(np.median(difference))

    output = {
        "analysis": "paired_layer_15_coordinate_state_restoration",
        "intervention": INTERVENTION,
        "count": len(rows),
        "rows": rows,
        "holo": {
            "median_restoration_nats": float(np.median(holo)),
            "mean_restoration_nats": float(np.mean(holo)),
            "positive_count": int(np.sum(holo > 0)),
        },
        "qwen": {
            "median_restoration_nats": float(np.median(qwen)),
            "mean_restoration_nats": float(np.mean(qwen)),
            "positive_count": int(np.sum(qwen > 0)),
        },
        "paired_holo_minus_qwen": {
            "median_nats": median_difference,
            "mean_nats": float(np.mean(difference)),
            "positive_count": int(np.sum(difference > 0)),
            "bootstrap_median_95_ci_nats": [ci_low, ci_high],
            "likelihood_ratio_factor": math.exp(median_difference),
            "likelihood_ratio_factor_95_ci": [math.exp(ci_low), math.exp(ci_high)],
            "likelihood_ratio_percent_increase": 100 * math.expm1(median_difference),
            "likelihood_ratio_percent_increase_95_ci": [100 * math.expm1(ci_low), 100 * math.expm1(ci_high)],
            "exact_one_sided_sign_flip_p_mean": _exact_sign_flip_p(difference),
            "exact_one_sided_sign_test_p": _exact_positive_sign_p(int(np.sum(difference > 0)), len(difference)),
        },
        "method": {
            "bootstrap": "paired nonparametric percentile bootstrap of the median",
            "randomization_test": "all 2^8 paired sign flips; one-sided alternative Holo > Qwen; statistic is mean paired difference",
            "interpretation": "Exponentiating a nat difference yields a multiplicative change in the target-versus-distractor sequence likelihood ratio.",
            "caveat": "Eight mirrored prompts from four synthetic image pairs; exploratory within-image causal panel, not an independent-image benchmark.",
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
