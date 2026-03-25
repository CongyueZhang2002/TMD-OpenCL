from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CandidateSpec, FITS_DIR
from deep_refit_current_best_42 import DeepBudgetFitSession

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "Cards"
RESULTS_DIR = FITS_DIR / "corrected_42_variant_refit_results"
RESULTS_DIR.mkdir(exist_ok=True)

CANDIDATES = [
    ("broad_bump_42_fixed_gauss_basis_corrected", "BroadBump42FixedGaussBasisBest"),
    ("poly_bstar_cslog_loggauss_42best_refresh", "PolyBstarCSLogLogGauss42Best"),
]


def parse_array(card_text: str, name: str):
    m = re.search(rf"{name}\s*=\s*\[(.*?)\]", card_text, re.S)
    if not m:
        raise RuntimeError(f"Missing {name}")
    return ast.literal_eval("[" + m.group(1) + "]")


def parse_param_names(card_text: str):
    m = re.search(r"struct Params_Struct\s*(.*?)end", card_text, re.S)
    if not m:
        raise RuntimeError("Missing Params_Struct")
    return re.findall(r"([A-Za-z_][A-Za-z0-9_]*)::Float32", m.group(1))


def load_spec(label: str, fit_name: str) -> CandidateSpec:
    card_path = CARDS_DIR / f"{fit_name}.jl"
    card_text = card_path.read_text(encoding="utf-8")
    np_name = re.search(r'const NP_name = "([^"]+)"', card_text).group(1)
    return CandidateSpec(
        name=label,
        fit_name=fit_name,
        np_name=np_name,
        param_names=parse_param_names(card_text),
        initial_params=[float(x) for x in parse_array(card_text, "initial_params")],
        bounds=[tuple(map(float, pair)) for pair in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="baseline",
    )


def run_one(label: str, fit_name: str, maxfun: int) -> dict:
    print(f"[candidate] {fit_name}", flush=True)
    spec = load_spec(label, fit_name)
    session = DeepBudgetFitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
    result = {
        "candidate": label,
        "fit_name": fit_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }
    out_path = RESULTS_DIR / f"{label}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = best_eval["metrics"]
    row = {
        "candidate": label,
        "fit_name": fit_name,
        "chi2dN_total": metrics["chi2dN_total"],
        "chi2dN_collider": metrics["chi2dN_collider"],
        "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
        "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
        "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
        "fit_evals": fit_info["nf"],
        "fit_elapsed_s": fit_info["elapsed_s"],
    }
    for name, value in zip(spec.param_names, best_eval["full_params"]):
        row[name] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxfun", type=int, default=420)
    args = parser.parse_args()

    rows = []
    for label, fit_name in CANDIDATES:
        rows.append(run_one(label, fit_name, maxfun=args.maxfun))
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "summary.csv", index=False)

    pd.DataFrame(rows).sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'summary.csv'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
