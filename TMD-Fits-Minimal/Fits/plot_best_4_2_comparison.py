from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
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
from aggressive_refit_table_models import (
    RESULTS_DIR as AGGR_RESULTS_DIR,
    MODELS as AGGR_MODELS,
    cleanup_generated_files as cleanup_aggr_generated_files,
    write_generated_files as write_aggr_generated_files,
)


OUT_DIR = FITS_DIR / "model_comparison_plots" / "best_4_2_comparison"
DATA_ROOT = FITS_DIR.parent / "Data" / "Default" / "Cutted" / "DY"

SHORTLIST = {
    "poly_bstar_cslog_loggauss": {
        "result_path": FITS_DIR / "localized_shape_followup_results" / "poly_bstar_cslog_loggauss.json",
        "label": "0-2 loggauss",
        "color": "tab:green",
        "linestyle": "-",
    },
    "reduced_loggauss_powerseed": {
        "result_path": FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json",
        "label": "0-2 powerseed",
        "color": "tab:blue",
        "linestyle": "--",
    },
    "poly_bstar_cslog__MSHT20N3LO-MC-4-2__aggressive": {
        "result_path": AGGR_RESULTS_DIR / "poly_bstar_cslog__4-2.json",
        "label": "4-2 poly-bstar",
        "color": "tab:orange",
        "linestyle": ":",
    },
    "poly_bstar_cslog_loggauss__MSHT20N3LO-MC-4-2__aggressive": {
        "result_path": AGGR_RESULTS_DIR / "poly_bstar_cslog_loggauss__4-2.json",
        "label": "4-2 best",
        "color": "tab:red",
        "linestyle": "-.",
    },
}

DEFAULT_FILES = [
    r"CMS_13\CMS13-170Q350.csv",
    r"CMS_13\CMS13-350Q1000.csv",
    r"ATLAS_8\ATLAS8-00y04.csv",
    r"ATLAS_8\ATLAS8-04y08.csv",
    r"LHCb_8\LHCb8.csv",
]


def parse_struct_fields(card_text: str) -> list[str]:
    match = re.search(r"struct\s+Params_Struct(.*?)end", card_text, re.S)
    if not match:
        raise RuntimeError("Could not find Params_Struct block")
    fields: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            fields.append(part.split("::", 1)[0].strip())
    return fields


def parse_last_bracket_block(text: str, name: str) -> str:
    pattern = rf"(?ms)^[ \t]*(?!#){re.escape(name)}\s*=\s*\[(.*?)\]"
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"Could not find {name}")
    return matches[-1]


def parse_array(text: str, name: str) -> list:
    raw = parse_last_bracket_block(text, name)
    cleaned = []
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            cleaned.append(stripped)
    joined = " ".join(cleaned)
    if not joined.strip():
        return []
    return list(ast.literal_eval("[" + joined + "]"))


def parse_scalar_string(text: str, name: str) -> str:
    match = re.search(rf'(?m)^[ \t]*(?!#){re.escape(name)}\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError(f"Could not find {name}")
    return match.group(1)


def make_card_spec(card_name: str, result_path: Path) -> object:
    from auto_np_search import CARDS_DIR, CandidateSpec

    card_path = CARDS_DIR / f"{card_name}.jl"
    card_text = card_path.read_text(encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return CandidateSpec(
        name=card_name,
        fit_name=card_name,
        np_name=parse_scalar_string(card_text, "const NP_name"),
        param_names=parse_struct_fields(card_text),
        initial_params=[float(x) for x in result["best"]["full_params"]],
        bounds=[(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="plot_card_spec",
    )


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
            style = SHORTLIST[candidate]
            pred_vals = cand_df["prediction"].to_numpy(dtype=float)
            ratio_vals = cand_df["ratio"].to_numpy(dtype=float)
            pred_arrays.append(pred_vals)
            ratio_arrays.append(ratio_vals)
            line_top, = ax_top.plot(
                qT,
                pred_vals,
                linewidth=1.7,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
            )
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


def load_specs() -> dict[str, object]:
    cleanup_aggr_generated_files()
    specs: dict[str, object] = {}
    specs["poly_bstar_cslog_loggauss"] = make_card_spec(
        "PolyBstarCSLogLogGauss",
        SHORTLIST["poly_bstar_cslog_loggauss"]["result_path"],
    )
    specs["reduced_loggauss_powerseed"] = make_card_spec(
        "ReducedLogGaussPowerSeed",
        SHORTLIST["reduced_loggauss_powerseed"]["result_path"],
    )

    for model_name in ["poly_bstar_cslog", "poly_bstar_cslog_loggauss"]:
        model = next(item for item in AGGR_MODELS if item.name == model_name)
        spec = write_aggr_generated_files(model, "MSHT20N3LO-MC-4-2", 4)
        specs[spec.name] = spec

    return specs


def export_candidate_rows(candidate: str, specs: dict[str, object], files: list[str], out_dir: Path) -> None:
    spec = specs[candidate]
    session = FitSession(spec)
    result = json.loads(Path(SHORTLIST[candidate]["result_path"]).read_text(encoding="utf-8"))
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
                    "label": SHORTLIST[candidate]["label"],
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
        "label": SHORTLIST[candidate]["label"],
        "fit_name": result.get("fit_name", candidate),
        "param_names": result["param_names"],
        "full_params": full_params.tolist(),
        "metrics": metrics,
    }
    (out_dir / f"{candidate}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_worker(candidate: str, files: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    try:
        export_candidate_rows(candidate, specs, files, out_dir)
    finally:
        cleanup_aggr_generated_files()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = build_default_file_list()

    for candidate in SHORTLIST:
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

    summary = summarize(out_dir, list(SHORTLIST))
    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_combined(out_dir, list(SHORTLIST), files, grid_filename="all_experiments_grid.png")
    plot_combined(out_dir, list(SHORTLIST), DEFAULT_FILES, grid_filename="high_energy_grid.png")

    rows = []
    for candidate in SHORTLIST:
        meta = json.loads((out_dir / f"{candidate}_meta.json").read_text(encoding="utf-8"))
        metrics = meta["metrics"]
        rows.append(
            {
                "candidate": candidate,
                "label": meta["label"],
                "chi2dN_total": metrics["chi2dN_total"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "best_4_2_summary.csv", index=False)


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
