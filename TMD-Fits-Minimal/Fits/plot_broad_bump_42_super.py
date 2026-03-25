from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from auto_np_search import EXPERIMENTS, FILE_EXCLUDES, FITS_DIR, FitSession
from deep_refit_corrected_42_variants import load_spec


OUT_DIR = FITS_DIR / "model_comparison_plots" / "broad_bump_42_super"
DATA_ROOT = FITS_DIR.parent / "Data" / "Default" / "Cutted" / "DY"

SPEC = load_spec("broad_bump_42_alpha1_nolambda2_super", "BroadBump42LogGaussAlpha1NoLambda2")

DEFAULT_FILES = [
    r"CMS_13\CMS13-170Q350.csv",
    r"CMS_13\CMS13-350Q1000.csv",
    r"ATLAS_8\ATLAS8-00y04.csv",
    r"ATLAS_8\ATLAS8-04y08.csv",
    r"LHCb_8\LHCb8.csv",
]


def build_default_file_list() -> list[str]:
    file_names: list[str] = []
    for experiment in EXPERIMENTS:
        exp_dir = DATA_ROOT / experiment
        for path in sorted(exp_dir.glob("*.csv")):
            rel = str(Path(experiment) / path.name)
            if rel.replace("\\", "/") in FILE_EXCLUDES:
                continue
            file_names.append(rel)
    return file_names


def export_bins(files: list[str]) -> tuple[pd.DataFrame, dict]:
    session = FitSession(SPEC)
    full_params = np.asarray(SPEC.initial_params, dtype=float)

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

    bins = pd.DataFrame(rows)
    meta = {
        "fit_name": SPEC.fit_name,
        "param_names": SPEC.param_names,
        "full_params": full_params.tolist(),
        "metrics": metrics,
    }
    return bins, meta


def _set_spectrum_ylim(ax, y_data, y_err, y_pred, pad_frac: float = 0.12) -> None:
    y_data = np.asarray(y_data, float)
    y_err = np.asarray(y_err, float)
    y_pred = np.asarray(y_pred, float)
    data_up = y_data + np.nan_to_num(y_err, nan=0.0)
    candidates = np.concatenate([data_up[np.isfinite(data_up)], y_pred[np.isfinite(y_pred)]])
    if candidates.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    ymax = float(np.max(candidates))
    if not np.isfinite(ymax) or ymax <= 0:
        ax.set_ylim(0.0, 1.0)
        return
    ax.set_ylim(0.0, np.nextafter(ymax * (1.0 + pad_frac), np.inf))


def _set_ratio_ylim(ax, ratio_vals, ratio_errs, margin_frac: float = 0.22, min_half_frac: float = 0.08) -> None:
    ratio_vals = np.asarray(ratio_vals, float)
    ratio_vals = ratio_vals[np.isfinite(ratio_vals)]
    ratio_errs = np.asarray(ratio_errs, float)
    ratio_errs = ratio_errs[np.isfinite(ratio_errs)]

    lower = float(np.min(ratio_vals)) if ratio_vals.size else 1.0
    upper = float(np.max(ratio_vals)) if ratio_vals.size else 1.0
    if ratio_errs.size:
        lower = min(lower, 1.0 - float(np.max(ratio_errs)))
        upper = max(upper, 1.0 + float(np.max(ratio_errs)))

    half_needed = max(1.0 - lower, upper - 1.0, 0.0)
    half = max(half_needed * (1.0 + margin_frac), min_half_frac)
    ax.set_ylim(1.0 - half, 1.0 + half)


def plot_combined(bins: pd.DataFrame, files: list[str], out_path: Path) -> None:
    n_files = len(files)
    ncols = min(3, n_files)
    nrow_pairs = math.ceil(n_files / ncols)
    fig = plt.figure(figsize=(4.9 * ncols, 4.0 * nrow_pairs))
    outer = fig.add_gridspec(nrows=nrow_pairs, ncols=ncols, wspace=0.42, hspace=0.34)

    hide_zero_label = FuncFormatter(lambda y, pos: "" if np.isclose(y, 0.0) else f"{y:g}")

    for idx, file_name in enumerate(files):
        r = idx // ncols
        c = idx % ncols
        inner = outer[r, c].subgridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)

        file_df = bins[bins["file"] == file_name].sort_values("bin_index")
        if file_df.empty:
            continue

        qT = file_df["qT"].to_numpy(dtype=float)
        data_vals = file_df["data"].to_numpy(dtype=float)
        data_err = file_df["error_data"].to_numpy(dtype=float)
        total_err = file_df["error_total"].to_numpy(dtype=float)
        pred_vals = file_df["prediction"].to_numpy(dtype=float)
        ratio_vals = file_df["ratio"].to_numpy(dtype=float)
        ratio_err_data = file_df["error_data_ratio"].to_numpy(dtype=float)
        ratio_err_total = file_df["error_total_ratio"].to_numpy(dtype=float)

        ax_top.errorbar(qT, data_vals, yerr=total_err, fmt="none", ecolor="gray", elinewidth=1, capsize=3)
        ax_top.errorbar(qT, data_vals, yerr=data_err, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor="dodgerblue")
        ax_top.plot(qT, pred_vals, linewidth=1.8, color="tab:red")
        ax_top.set_title(Path(file_name).stem)
        ax_top.set_ylabel("dσ/dqT")
        ax_top.grid(True, alpha=0.3)
        _set_spectrum_ylim(ax_top, data_vals, np.maximum(data_err, total_err), pred_vals)
        ax_top.yaxis.set_major_formatter(hide_zero_label)
        ax_top.tick_params(labelbottom=False)

        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_total, fmt="none", ecolor="gray", elinewidth=1, capsize=3)
        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_data, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor="dodgerblue")
        ax_bot.plot(qT, ratio_vals, linewidth=1.6, color="tab:red")
        ax_bot.axhline(1.0, color="black", linewidth=1.0, alpha=0.75)
        ax_bot.set_xlabel("qT")
        ax_bot.set_ylabel("pred/data")
        ax_bot.grid(True, alpha=0.3)
        _set_ratio_ylim(ax_bot, ratio_vals, np.maximum(ratio_err_data, ratio_err_total))

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = build_default_file_list()
    bins, meta = export_bins(files)
    bins.to_csv(OUT_DIR / "bins.csv", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame([{"fit_name": meta["fit_name"], **meta["metrics"]}]).to_csv(OUT_DIR / "summary.csv", index=False)

    plot_combined(bins, files, OUT_DIR / "all_experiments_grid.png")
    plot_combined(bins, DEFAULT_FILES, OUT_DIR / "high_energy_grid.png")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
