from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from auto_np_search import EXPERIMENTS, FILE_EXCLUDES, FITS_DIR, PYTHON, FitSession
from parameter_significance_0_2 import NestedFit
from parameter_significance_good_variants import MODELS as GOOD_MODELS, make_spec as make_good_spec


OUT_DIR = FITS_DIR / "model_comparison_plots" / "powerseed_lambda4_comparison"
BASE_RESULT = FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json"

CANDIDATES = {
    "powerseed_full": {
        "label": "Powerseed",
        "color": "tab:green",
        "linestyle": "-",
    },
    "powerseed_lambda4_zero": {
        "label": "Powerseed, lambda4=0",
        "color": "tab:red",
        "linestyle": "--",
    },
}

DEFAULT_FILES = [
    r"CMS_13\CMS13-170Q350.csv",
    r"CMS_13\CMS13-350Q1000.csv",
    r"ATLAS_8\ATLAS8-00y04.csv",
    r"ATLAS_8\ATLAS8-04y08.csv",
    r"LHCb_8\LHCb8.csv",
]


def build_default_file_list() -> list[str]:
    file_root = FITS_DIR.parent / "Data" / "Default" / "Cutted" / "DY"
    file_names: list[str] = []
    for experiment in EXPERIMENTS:
        exp_dir = file_root / experiment
        for path in sorted(exp_dir.glob("*.csv")):
            rel = str(Path(experiment) / path.name)
            if rel.replace("\\", "/") in FILE_EXCLUDES:
                continue
            file_names.append(rel)
    return file_names


def summarize(out_dir: Path, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        meta_path = out_dir / f"{candidate}_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta["metrics"]
        rows.append(
            {
                "candidate": candidate,
                "label": meta["label"],
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2dN_collider": metrics["chi2dN_collider"],
                "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
            }
        )
    return pd.DataFrame(rows).sort_values("chi2dN_total").reset_index(drop=True)


def _set_spectrum_ylim(ax, y_data, y_err, y_preds, pad_frac: float = 0.12) -> None:
    y_data = np.asarray(y_data, float)
    y_err = np.asarray(y_err, float)
    pred_arrays = [np.asarray(arr, float) for arr in y_preds]
    data_up = y_data + np.nan_to_num(y_err, nan=0.0)
    finite_parts = [data_up[np.isfinite(data_up)]]
    for arr in pred_arrays:
        finite_parts.append(arr[np.isfinite(arr)])
    candidates = np.concatenate([arr for arr in finite_parts if arr.size > 0])
    if candidates.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    ymax = float(np.max(candidates))
    if not np.isfinite(ymax) or ymax <= 0:
        ax.set_ylim(0.0, 1.0)
        return
    ax.set_ylim(0.0, np.nextafter(ymax * (1.0 + pad_frac), np.inf))


def _set_ratio_ylim(ax, ratio_arrays, ratio_errs, margin_frac: float = 0.22, min_half_frac: float = 0.08) -> None:
    lower = np.inf
    upper = -np.inf
    for arr in ratio_arrays:
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lower = min(lower, float(np.min(arr)))
            upper = max(upper, float(np.max(arr)))
    ratio_errs = np.asarray(ratio_errs, float)
    ratio_errs = ratio_errs[np.isfinite(ratio_errs)]
    if ratio_errs.size:
        lower = min(lower, 1.0 - float(np.max(ratio_errs)))
        upper = max(upper, 1.0 + float(np.max(ratio_errs)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        ax.set_ylim(0.9, 1.1)
        return
    half_needed = max(1.0 - lower, upper - 1.0, 0.0)
    half = max(half_needed * (1.0 + margin_frac), min_half_frac)
    ax.set_ylim(1.0 - half, 1.0 + half)


def plot_combined(out_dir: Path, candidates: list[str], files: list[str], grid_filename: str) -> None:
    bins_frames = []
    for candidate in candidates:
        bins_path = out_dir / f"{candidate}_bins.csv"
        if bins_path.exists():
            bins_frames.append(pd.read_csv(bins_path))
    if not bins_frames:
        raise RuntimeError("No exported bin data found.")
    bins = pd.concat(bins_frames, ignore_index=True)

    n_files = len(files)
    ncols = min(3, n_files)
    nrow_pairs = math.ceil(n_files / ncols)
    fig = plt.figure(figsize=(4.9 * ncols, 4.0 * nrow_pairs))
    outer = fig.add_gridspec(nrows=nrow_pairs, ncols=ncols, wspace=0.42, hspace=0.34)

    hide_zero_label = FuncFormatter(lambda y, pos: "" if np.isclose(y, 0.0) else f"{y:g}")
    legend_handles = []
    legend_labels = []

    for idx, file_name in enumerate(files):
        r = idx // ncols
        c = idx % ncols
        inner = outer[r, c].subgridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)

        file_df = bins[bins["file"] == file_name].copy()
        if file_df.empty:
            continue

        ref = file_df[file_df["candidate"] == candidates[0]].sort_values("bin_index")
        qT = ref["qT"].to_numpy(dtype=float)
        data_vals = ref["data"].to_numpy(dtype=float)
        data_err = ref["error_data"].to_numpy(dtype=float)
        total_err = ref["error_total"].to_numpy(dtype=float)
        ratio_err_data = ref["error_data_ratio"].to_numpy(dtype=float)
        ratio_err_total = ref["error_total_ratio"].to_numpy(dtype=float)

        ax_top.errorbar(qT, data_vals, yerr=total_err, fmt="none", ecolor="gray", elinewidth=1, capsize=3)
        ax_top.errorbar(qT, data_vals, yerr=data_err, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor="dodgerblue")

        pred_arrays = []
        ratio_arrays = []
        for candidate in candidates:
            cand_df = file_df[file_df["candidate"] == candidate].sort_values("bin_index")
            if cand_df.empty:
                continue
            style = CANDIDATES[candidate]
            pred_vals = cand_df["prediction"].to_numpy(dtype=float)
            ratio_vals = cand_df["ratio"].to_numpy(dtype=float)
            pred_arrays.append(pred_vals)
            ratio_arrays.append(ratio_vals)
            line_top, = ax_top.plot(qT, pred_vals, linewidth=1.7, color=style["color"], linestyle=style["linestyle"], label=style["label"])
            ax_bot.plot(qT, ratio_vals, linewidth=1.5, color=style["color"], linestyle=style["linestyle"])
            if idx == 0:
                legend_handles.append(line_top)
                legend_labels.append(style["label"])

        ax_top.set_title(Path(file_name).stem)
        ax_top.set_ylabel("dσ/dqT")
        ax_top.grid(True, alpha=0.3)
        _set_spectrum_ylim(ax_top, data_vals, np.maximum(data_err, total_err), pred_arrays)
        ax_top.yaxis.set_major_formatter(hide_zero_label)
        ax_top.tick_params(labelbottom=False)

        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_total, fmt="none", ecolor="gray", elinewidth=1, capsize=3)
        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_data, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor="dodgerblue")
        ax_bot.axhline(1.0, color="black", linewidth=1.0, alpha=0.75)
        ax_bot.set_xlabel("qT")
        ax_bot.set_ylabel("pred/data")
        ax_bot.grid(True, alpha=0.3)
        _set_ratio_ylim(ax_bot, ratio_arrays, np.maximum(ratio_err_data, ratio_err_total))

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_dir / grid_filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def powerseed_model():
    return next(model for model in GOOD_MODELS if model.name == "reduced_loggauss_powerseed")


def export_candidate_rows(candidate: str, files: list[str], out_dir: Path) -> None:
    model = powerseed_model()
    spec = make_good_spec(model)
    session = FitSession(spec)

    if candidate == "powerseed_full":
        base = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
        full_params = np.asarray(base["best"]["full_params"], dtype=float)
    elif candidate == "powerseed_lambda4_zero":
        best_full = np.asarray(session.initial_params, dtype=float)
        nested = NestedFit(session, best_full, {"lambda4": 0.0})
        fit_info = nested.solve(maxfun=90)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        full_params = np.asarray(best_nested["full_params"], dtype=float)
    else:
        raise ValueError(candidate)

    predictions, compute_s = session._predict(full_params)
    _, chi2_list, n_list = session.get_chi2(predictions)
    metrics = session.summary_metrics(predictions, chi2_list, compute_s)

    rows: list[dict[str, float | int | str]] = []
    for file_name in files:
        if file_name not in predictions:
            continue
        df_data = session.data_list[file_name].reset_index(drop=True)
        pred = np.asarray(predictions[file_name], dtype=float)
        data = df_data["xsec"].to_numpy(dtype=float)
        qT = df_data["qT_mean"].to_numpy(dtype=float)
        ratio = pred / data

        matrix_data = pd.read_csv(session.matrix_root / file_name).to_numpy(dtype=float)
        if session.data_uncertainty_only:
            matrix_total = matrix_data
        else:
            matrix_pdf = pd.read_csv(session.error_sets_root / file_name).to_numpy(dtype=float)
            matrix_total = matrix_data + matrix_pdf

        error_data = np.sqrt(np.diag(matrix_data))
        error_total = np.sqrt(np.diag(matrix_total))
        error_data_ratio = error_data / data
        error_total_ratio = error_total / data

        for i in range(len(qT)):
            rows.append(
                {
                    "candidate": candidate,
                    "label": CANDIDATES[candidate]["label"],
                    "file": file_name,
                    "bin_index": i,
                    "qT": float(qT[i]),
                    "data": float(data[i]),
                    "prediction": float(pred[i]),
                    "ratio": float(ratio[i]),
                    "error_data": float(error_data[i]),
                    "error_total": float(error_total[i]),
                    "error_data_ratio": float(error_data_ratio[i]),
                    "error_total_ratio": float(error_total_ratio[i]),
                    "chi2dN_file": float(chi2_list[file_name]),
                    "N_file": int(n_list[file_name]),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / f"{candidate}_bins.csv", index=False)
    meta = {
        "candidate": candidate,
        "label": CANDIDATES[candidate]["label"],
        "fit_name": spec.fit_name,
        "param_names": spec.param_names,
        "full_params": full_params.tolist(),
        "metrics": metrics,
    }
    (out_dir / f"{candidate}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_worker(candidate: str, files: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_candidate_rows(candidate, files, out_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = build_default_file_list()

    for candidate in CANDIDATES:
        cmd = [
            str(PYTHON),
            str(Path(__file__)),
            "--worker",
            "--candidate",
            candidate,
            "--outdir",
            str(out_dir),
        ]
        print(f"\n=== Exporting {candidate} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{candidate}: exit code {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {candidate}")

    summary = summarize(out_dir, list(CANDIDATES))
    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_combined(out_dir, list(CANDIDATES), files, "all_experiments_grid.png")
    plot_combined(out_dir, list(CANDIDATES), DEFAULT_FILES, "high_energy_grid.png")
    summary.to_csv(out_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", type=str)
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    if args.worker:
        if not args.candidate:
            raise ValueError("--worker requires --candidate")
        run_worker(args.candidate, build_default_file_list(), out_dir)
        return

    orchestrate(out_dir)


if __name__ == "__main__":
    main()
