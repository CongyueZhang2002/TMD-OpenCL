from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import PYTHON, FitSession
from scan_table_variants import (
    BASE_MODELS,
    RESULTS_DIR as LOCAL_SCAN_RESULTS_DIR,
    TABLE_VARIANTS,
    cleanup_generated_files,
    write_generated_files,
)


ROOT = Path(__file__).resolve().parents[1]
FITS_DIR = ROOT / "Fits"
RESULTS_DIR = FITS_DIR / "table_variant_refit_results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_combo(model_name: str, table_name: str, maxfun: int) -> Path:
    model = next(item for item in BASE_MODELS if item.name == model_name)
    _, n, m = next(item for item in TABLE_VARIANTS if item[0] == table_name)
    spec = write_generated_files(model, table_name, n)

    session = FitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "model": model.name,
        "label": model.label,
        "table_name": table_name,
        "n": n,
        "m": m,
        "fit_name": spec.fit_name,
        "np_name": spec.np_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }

    out_path = RESULTS_DIR / f"{model.name}__{table_name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        initial_metrics = data["initial"]["metrics"]
        best_metrics = data["best"]["metrics"]
        rows.append(
            {
                "model": data["model"],
                "label": data["label"],
                "table_name": data["table_name"],
                "n": data["n"],
                "m": data["m"],
                "initial_chi2dN_total": initial_metrics["chi2dN_total"],
                "chi2dN_total": best_metrics["chi2dN_total"],
                "chi2dN_collider": best_metrics["chi2dN_collider"],
                "chi2dN_fixed_target": best_metrics["chi2dN_fixed_target"],
                "highE_weighted_chi2dN": best_metrics["highE_weighted_chi2dN"],
                "highE_mean_absdev_first3": best_metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": best_metrics["highE_mean_shortfall_first3"],
                "cms_highmass_mean_signed_first3": best_metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": best_metrics["zlike_mean_signed_first3"],
                "delta_chi2_vs_initial": best_metrics["chi2dN_total"] - initial_metrics["chi2dN_total"],
                "delta_absdev_vs_initial": best_metrics["highE_mean_absdev_first3"] - initial_metrics["highE_mean_absdev_first3"],
                "fit_evals": data["fit"]["nf"],
                "fit_elapsed_s": data["fit"]["elapsed_s"],
                "stage1_best_log10_chi2": min(item["log10_chi2"] for item in data["fit"]["stage1"]),
                "stage2_best_log10_chi2": min(item["log10_chi2"] for item in data["fit"]["stage2"]),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["model", "highE_mean_absdev_first3", "chi2dN_total"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    lines = [
        "# Deep Refit Table Variant Scan",
        "",
        f"Local pre-scan reference: `{LOCAL_SCAN_RESULTS_DIR / 'summary.csv'}`",
        "",
    ]

    for model_name in df["model"].drop_duplicates():
        subset = df[df["model"] == model_name]
        best_abs = subset.sort_values(["highE_mean_absdev_first3", "chi2dN_total"]).iloc[0]
        best_short = subset.sort_values(["highE_mean_shortfall_first3", "chi2dN_total"]).iloc[0]
        best_chi2 = subset.sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).iloc[0]
        lines.append(
            f"- `{model_name}`: best chi2 `{best_chi2['table_name']}` ({best_chi2['chi2dN_total']:.6f}), "
            f"best absdev `{best_abs['table_name']}` ({best_abs['highE_mean_absdev_first3']:.6f}), "
            f"best shortfall `{best_short['table_name']}` ({best_short['highE_mean_shortfall_first3']:.6f})"
        )
    lines.append("")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))


def orchestrate(model_names: list[str], table_names: list[str], maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for model_name in model_names:
            for table_name in table_names:
                cmd = [
                    str(PYTHON),
                    str(Path(__file__)),
                    "--combo",
                    model_name,
                    table_name,
                    "--maxfun",
                    str(maxfun),
                ]
                print(f"\n=== Deep refit {model_name} on {table_name} ===")
                proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
                print(f"exit code {proc.returncode}")
        print("\n=== Summary ===")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", nargs=2, metavar=("MODEL", "TABLE"))
    parser.add_argument("--models", nargs="*", default=["poly_bstar_cslog"])
    parser.add_argument("--tables", nargs="*", default=[name for name, _, _ in TABLE_VARIANTS])
    parser.add_argument("--maxfun", type=int, default=180)
    args = parser.parse_args()

    if args.combo:
        model_name, table_name = args.combo
        out_path = run_combo(model_name, table_name, maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(args.models, args.tables, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
