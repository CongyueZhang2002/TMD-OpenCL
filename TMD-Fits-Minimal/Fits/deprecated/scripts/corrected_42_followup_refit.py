from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from deep_refit_corrected_42_variants import load_spec
from deep_refit_current_best_42 import DeepBudgetFitSession


def run_one(label: str, fit_name: str, maxfun: int, results_dir: Path) -> dict:
    print(f"[candidate] {fit_name}", flush=True)
    spec = load_spec(label, fit_name)
    session = DeepBudgetFitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(fit_info["free_params"])

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

    out_path = results_dir / f"{label}.json"
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

    row_path = results_dir / f"{label}.row.json"
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--fit-name", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--maxfun", type=int, default=420)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    row = run_one(args.label, args.fit_name, args.maxfun, results_dir)

    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["candidate"] != row["candidate"]]
        summary = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    summary = summary.sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
