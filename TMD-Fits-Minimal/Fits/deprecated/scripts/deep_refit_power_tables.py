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
from scan_table_variants import BASE_MODELS, cleanup_generated_files, write_generated_files


ROOT = Path(__file__).resolve().parents[1]
FITS_DIR = ROOT / "Fits"
RESULTS_DIR = FITS_DIR / "power_table_refit_results"
RESULTS_DIR.mkdir(exist_ok=True)

# These power-correction tables were introduced as alternatives to 0-2.
# Treat them with the same runtime mustar choice as n=0 unless/until
# separate table-generation metadata says otherwise.
POWER_TABLES = [
    ("MSHT20N3LO-MC-0-2", 0, 2, "baseline"),
    ("MSHT20N3LO-MC-power-full", 0, 2, "power_full"),
    ("MSHT20N3LO-MC-power-x", 0, 2, "power_x"),
    ("MSHT20N3LO-MC-power-leptonic", 0, 2, "power_leptonic"),
]


def run_combo(model_name: str, table_name: str, maxfun: int) -> Path:
    model = next(item for item in BASE_MODELS if item.name == model_name)
    _, n, m, tag = next(item for item in POWER_TABLES if item[0] == table_name)
    spec = write_generated_files(model, table_name, n)

    session = FitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "model": model.name,
        "label": model.label,
        "table_name": table_name,
        "table_tag": tag,
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
                "table_tag": data["table_tag"],
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
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "highE_mean_absdev_first3", "chi2dN_total"]).reset_index(drop=True)

    baseline_rows = []
    for model_name in df["model"].drop_duplicates():
        subset = df[df["model"] == model_name].copy()
        base = subset[subset["table_name"] == "MSHT20N3LO-MC-0-2"].iloc[0]
        for idx, row in subset.iterrows():
            df.loc[idx, "delta_total_vs_0_2"] = row["chi2dN_total"] - base["chi2dN_total"]
            df.loc[idx, "delta_absdev_vs_0_2"] = row["highE_mean_absdev_first3"] - base["highE_mean_absdev_first3"]
            df.loc[idx, "delta_shortfall_vs_0_2"] = row["highE_mean_shortfall_first3"] - base["highE_mean_shortfall_first3"]
        best_abs = subset.sort_values(["highE_mean_absdev_first3", "chi2dN_total"]).iloc[0]
        best_short = subset.sort_values(["highE_mean_shortfall_first3", "chi2dN_total"]).iloc[0]
        best_chi2 = subset.sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).iloc[0]
        baseline_rows.append(
            {
                "model": model_name,
                "best_chi2_table": best_chi2["table_name"],
                "best_chi2dN_total": best_chi2["chi2dN_total"],
                "best_absdev_table": best_abs["table_name"],
                "best_absdev": best_abs["highE_mean_absdev_first3"],
                "best_shortfall_table": best_short["table_name"],
                "best_shortfall": best_short["highE_mean_shortfall_first3"],
            }
        )

    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(RESULTS_DIR / "best_by_model.csv", index=False)

    agg = (
        df.groupby("table_name", as_index=False)[
            ["chi2dN_total", "highE_mean_absdev_first3", "highE_mean_shortfall_first3"]
        ]
        .mean()
        .sort_values(["highE_mean_absdev_first3", "chi2dN_total"])
        .reset_index(drop=True)
    )
    agg.to_csv(RESULTS_DIR / "aggregate_means.csv", index=False)

    lines = [
        "# Power Table Refit Scan",
        "",
        "Assumption: all power tables use the same runtime `mustar_func` choice as `MSHT20N3LO-MC-0-2`, i.e. `n=0`.",
        "",
        "Scanned models:",
        "",
    ]
    for model in BASE_MODELS:
        lines.append(f"- `{model.name}`: {model.label}")
    lines.extend(["", "Tables scanned:", ""])
    for table_name, n, m, tag in POWER_TABLES:
        lines.append(f"- `{table_name}` (`{tag}`), runtime `n={n}`, reference `m={m}`")
    lines.extend(["", "Best table per model:", ""])
    for row in baseline_rows:
        lines.append(
            f"- `{row['model']}`: best chi2 `{row['best_chi2_table']}` ({row['best_chi2dN_total']:.6f}), "
            f"best absdev `{row['best_absdev_table']}` ({row['best_absdev']:.6f}), "
            f"best shortfall `{row['best_shortfall_table']}` ({row['best_shortfall']:.6f})"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print("\nAggregate means:")
    print(agg.to_string(index=False))


def orchestrate(model_names: list[str], maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for model_name in model_names:
            for table_name, _, _, _ in POWER_TABLES:
                cmd = [
                    str(PYTHON),
                    str(Path(__file__)),
                    "--combo",
                    model_name,
                    table_name,
                    "--maxfun",
                    str(maxfun),
                ]
                print(f"\n=== Power refit {model_name} on {table_name} ===")
                proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
                print(f"exit code {proc.returncode}")
        print("\n=== Summary ===")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", nargs=2, metavar=("MODEL", "TABLE"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["baseline_unfrozen", "poly_bstar_cslog", "poly_bstar_cslog_loggauss"],
    )
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
        orchestrate(args.models, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
