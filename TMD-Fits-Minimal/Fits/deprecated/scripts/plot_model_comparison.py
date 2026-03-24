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

from auto_np_search import EXPERIMENTS, FILE_EXCLUDES
from market_np_search import MarketFitSession, all_specs, write_auto_candidate_files


ROOT = Path(__file__).resolve().parents[1]
FITS_DIR = ROOT / "Fits"
RESULTS_DIR = FITS_DIR / "market_np_results"
OUT_ROOT = FITS_DIR / "model_comparison_plots"
PYTHON = Path(sys.executable)

DEFAULT_CANDIDATES = [
    "baseline_0112",
    "art23_fi_cslog",
    "hybrid_0112_art23cs",
]

DEFAULT_FILES = [
    r"CMS_13\CMS13-170Q350.csv",
    r"CMS_13\CMS13-350Q1000.csv",
    r"ATLAS_8\ATLAS8-00y04.csv",
    r"ATLAS_8\ATLAS8-04y08.csv",
    r"LHCb_8\LHCb8.csv",
]

CANDIDATE_STYLE = {
    "baseline_0112": {"label": "0112", "color": "tab:orange", "linestyle": "-"},
    "art23_fi_cslog": {"label": "ART23 FI+CSlog", "color": "tab:blue", "linestyle": "--"},
    "hybrid_0112_art23cs": {"label": "0112 + ART23 CS", "color": "tab:green", "linestyle": "-."},
    "baseline_unfrozen": {"label": "0112 unfrozen", "color": "tab:red", "linestyle": ":"},
}


def result_path(candidate: str) -> Path:
    return RESULTS_DIR / f"{candidate}.json"


def slugify_file(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_")


def build_default_file_list() -> list[str]:
    file_root = ROOT / "Data" / "Default" / "Cutted" / "DY"
    file_names: list[str] = []
    for experiment in EXPERIMENTS:
        exp_dir = file_root / experiment
        for path in sorted(exp_dir.glob("*.csv")):
            rel = str(Path(experiment) / path.name)
            if rel.replace("\\", "/") in FILE_EXCLUDES:
                continue
            file_names.append(rel)
    return file_names


def reconstruct_full_params(result: dict) -> np.ndarray:
    full = result.get("best", {}).get("full_params")
    if full is not None:
        return np.asarray(full, dtype=float)

    full = result.get("fit", {}).get("full_params")
    if full is not None:
        return np.asarray(full, dtype=float)

    initial_full = np.asarray(result["initial"]["full_params"], dtype=float)
    free_params = np.asarray(result["fit"]["free_params"], dtype=float)
    free_idx = np.asarray(result["free_idx"], dtype=int)
    frozen_idx = np.asarray(result["frozen_idx"], dtype=int)

    full = initial_full.copy()
    full[free_idx] = free_params
    if frozen_idx.size:
        full[frozen_idx] = initial_full[frozen_idx]
    return full


def export_candidate_rows(candidate: str, files: list[str], out_dir: Path) -> Path:
    specs = all_specs()
    spec = specs[candidate]
    session = MarketFitSession(spec)

    result = json.loads(result_path(candidate).read_text(encoding="utf-8"))
    full_params = reconstruct_full_params(result)

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
                    "label": CANDIDATE_STYLE.get(candidate, {}).get("label", candidate),
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

    candidate_out = out_dir / f"{candidate}_bins.csv"
    pd.DataFrame(rows).to_csv(candidate_out, index=False)

    meta = {
        "candidate": candidate,
        "label": CANDIDATE_STYLE.get(candidate, {}).get("label", candidate),
        "fit_name": result["fit_name"],
        "param_names": result["param_names"],
        "full_params": full_params.tolist(),
        "metrics": metrics,
    }
    (out_dir / f"{candidate}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return candidate_out


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
    df = pd.DataFrame(rows).sort_values("chi2dN_total").reset_index(drop=True)
    df.to_csv(out_dir / "comparison_summary.csv", index=False)
    return df


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
        half = max(min_half_frac, 0.10)
        ax.set_ylim(1.0 - half, 1.0 + half)
        return

    half_needed = max(1.0 - lower, upper - 1.0, 0.0)
    half = max(half_needed * (1.0 + margin_frac), min_half_frac)
    ax.set_ylim(1.0 - half, 1.0 + half)


def plot_combined(
    out_dir: Path,
    candidates: list[str],
    files: list[str],
    write_individual: bool = True,
    annotate_chi2: bool = False,
    grid_filename: str = "comparison_grid.png",
) -> None:
    bins_frames = []
    meta = {}
    for candidate in candidates:
        bins_path = out_dir / f"{candidate}_bins.csv"
        meta_path = out_dir / f"{candidate}_meta.json"
        if not bins_path.exists() or not meta_path.exists():
            continue
        bins_frames.append(pd.read_csv(bins_path))
        meta[candidate] = json.loads(meta_path.read_text(encoding="utf-8"))

    if not bins_frames:
        raise RuntimeError("No exported bin data found.")

    bins = pd.concat(bins_frames, ignore_index=True)

    n_files = len(files)
    ncols = min(3, n_files)
    nrow_pairs = math.ceil(n_files / ncols)

    fig = plt.figure(figsize=(4.9 * ncols, 4.0 * nrow_pairs))
    outer = fig.add_gridspec(nrows=nrow_pairs, ncols=ncols, wspace=0.42, hspace=0.34)

    hide_zero_label = FuncFormatter(lambda y, pos: "" if np.isclose(y, 0.0) else f"{y:g}")

    data_color = "dodgerblue"
    total_error_color = "gray"

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

        first_candidate = candidates[0]
        ref = file_df[file_df["candidate"] == first_candidate].sort_values("bin_index")
        qT = ref["qT"].to_numpy(dtype=float)
        data_vals = ref["data"].to_numpy(dtype=float)
        data_err = ref["error_data"].to_numpy(dtype=float)
        total_err = ref["error_total"].to_numpy(dtype=float)
        ratio_err_data = ref["error_data_ratio"].to_numpy(dtype=float)
        ratio_err_total = ref["error_total_ratio"].to_numpy(dtype=float)

        ax_top.errorbar(qT, data_vals, yerr=total_err, fmt="none", ecolor=total_error_color, elinewidth=1, capsize=3)
        ax_top.errorbar(qT, data_vals, yerr=data_err, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor=data_color)

        pred_arrays = []
        ratio_arrays = []
        for candidate in candidates:
            cand_df = file_df[file_df["candidate"] == candidate].sort_values("bin_index")
            if cand_df.empty:
                continue
            style = CANDIDATE_STYLE.get(candidate, {})
            pred_vals = cand_df["prediction"].to_numpy(dtype=float)
            ratio_vals = cand_df["ratio"].to_numpy(dtype=float)
            pred_arrays.append(pred_vals)
            ratio_arrays.append(ratio_vals)

            line_top, = ax_top.plot(
                qT,
                pred_vals,
                linewidth=1.7,
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                label=style.get("label", candidate),
            )
            ax_bot.plot(
                qT,
                ratio_vals,
                linewidth=1.5,
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
            )
            if idx == 0:
                legend_handles.append(line_top)
                legend_labels.append(style.get("label", candidate))

        ax_top.set_title(Path(file_name).stem)
        ax_top.set_ylabel("dσ/dqT")
        ax_top.grid(True, alpha=0.3)
        _set_spectrum_ylim(ax_top, data_vals, np.maximum(data_err, total_err), pred_arrays)
        ax_top.yaxis.set_major_formatter(hide_zero_label)
        ax_top.tick_params(labelbottom=False)
        ax_top.spines["bottom"].set_visible(True)

        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_total, fmt="none", ecolor=total_error_color, elinewidth=1, capsize=3)
        ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_data, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor=data_color)
        ax_bot.axhline(1.0, color="black", linewidth=1.0, alpha=0.75)
        ax_bot.set_xlabel("qT")
        ax_bot.set_ylabel("pred/data")
        ax_bot.grid(True, alpha=0.3)
        _set_ratio_ylim(ax_bot, ratio_arrays, np.maximum(ratio_err_data, ratio_err_total))

        if annotate_chi2:
            chi2_bits = []
            for candidate in candidates:
                cand_df = file_df[file_df["candidate"] == candidate]
                if cand_df.empty:
                    continue
                label = CANDIDATE_STYLE.get(candidate, {}).get("label", candidate)
                chi2_bits.append(f"{label}: {cand_df['chi2dN_file'].iloc[0]:.2f}")
            ax_bot.text(
                0.02,
                0.08,
                "\n".join(chi2_bits),
                transform=ax_bot.transAxes,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
            )

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_dir / grid_filename, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if write_individual:
        for file_name in files:
            file_df = bins[bins["file"] == file_name].copy()
            if file_df.empty:
                continue
            fig = plt.figure(figsize=(5.3, 4.2))
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0)
            ax_top = fig.add_subplot(gs[0, 0])
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

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
                style = CANDIDATE_STYLE.get(candidate, {})
                pred_vals = cand_df["prediction"].to_numpy(dtype=float)
                ratio_vals = cand_df["ratio"].to_numpy(dtype=float)
                pred_arrays.append(pred_vals)
                ratio_arrays.append(ratio_vals)
                ax_top.plot(qT, pred_vals, linewidth=1.7, color=style.get("color"), linestyle=style.get("linestyle", "-"), label=style.get("label", candidate))
                ax_bot.plot(qT, ratio_vals, linewidth=1.5, color=style.get("color"), linestyle=style.get("linestyle", "-"))

            ax_top.set_title(Path(file_name).stem)
            ax_top.set_ylabel("dσ/dqT")
            ax_top.grid(True, alpha=0.3)
            _set_spectrum_ylim(ax_top, data_vals, np.maximum(data_err, total_err), pred_arrays)
            ax_top.legend(frameon=False, fontsize=9)
            ax_top.tick_params(labelbottom=False)

            ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_total, fmt="none", ecolor="gray", elinewidth=1, capsize=3)
            ax_bot.errorbar(qT, np.ones_like(qT), yerr=ratio_err_data, fmt="o", mfc="none", ms=2, elinewidth=1, capsize=2, ecolor="dodgerblue")
            ax_bot.axhline(1.0, color="black", linewidth=1.0, alpha=0.75)
            ax_bot.set_xlabel("qT")
            ax_bot.set_ylabel("pred/data")
            ax_bot.grid(True, alpha=0.3)
            _set_ratio_ylim(ax_bot, ratio_arrays, np.maximum(ratio_err_data, ratio_err_total))

            fig.tight_layout()
            fig.savefig(out_dir / f"{slugify_file(file_name)}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def run_worker(candidate: str, files: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_candidate_rows(candidate, files, out_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(candidates: list[str], files: list[str], out_dir: Path, write_individual: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_auto_candidate_files()
    all_specs()

    for candidate in candidates:
        cmd = [
            str(PYTHON),
            str(Path(__file__)),
            "--worker",
            "--candidate",
            candidate,
            "--outdir",
            str(out_dir),
            "--files",
            *files,
        ]
        print(f"\n=== Exporting {candidate} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{candidate}: exit code {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {candidate}")

    summary = summarize(out_dir, candidates)
    print("\nSummary:")
    if not summary.empty:
        print(summary.to_string(index=False))
    plot_combined(out_dir, candidates, files, write_individual=write_individual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--outdir", type=str, default=str(OUT_ROOT / "high_energy"))
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    files = build_default_file_list() if args.all_files else args.files
    if args.worker:
        if not args.candidate:
            raise ValueError("--worker requires --candidate")
        run_worker(args.candidate, files, out_dir)
        return

    orchestrate(args.candidates, files, out_dir, write_individual=not args.skip_individual)


if __name__ == "__main__":
    main()
