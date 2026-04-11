from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CandidateSpec, FITS_DIR, FitSession
from parameter_significance_0_2 import (
    NestedFit,
    finite_diff_hessian,
    parse_array,
    parse_struct_fields,
    psd_covariance,
    top_correlations,
)


REFERENCE_MAP = {
    "lambda1": 0.0,
    "lambda2": 0.0,
    "lambda3": 0.0,
    "lambda4": 0.0,
    "alpha": 1.0,
    "amp": 0.0,
    "c0": 0.0,
    "c1": 0.0,
}


def load_best_full(result_path: Path) -> list[float]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def make_spec(name: str, fit_name: str, card_path: Path, result_path: Path) -> CandidateSpec:
    card_text = card_path.read_text(encoding="utf-8")
    return CandidateSpec(
        name=name,
        fit_name=fit_name,
        np_name=card_text.split('const NP_name = "', 1)[1].split('"', 1)[0],
        param_names=parse_struct_fields(card_text),
        initial_params=load_best_full(result_path),
        bounds=[(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="single_candidate_significance",
    )


def write_temp_card(source_card: Path, temp_fit_name: str, full_params: list[float]) -> Path:
    card_text = source_card.read_text(encoding="utf-8")
    init_vals = ", ".join(f"{x:.10g}" for x in full_params)
    card_text = re.sub(
        r"initial_params\s*=\s*\[(.*?)\]",
        f"initial_params = [{init_vals}]",
        card_text,
        count=1,
        flags=re.S,
    )
    temp_card = source_card.parent / f"{temp_fit_name}.jl"
    temp_card.write_text(card_text, encoding="utf-8")
    return temp_card


def null_tests_for(best_full: list[float], spec: CandidateSpec) -> list[tuple[str, dict[str, float]]]:
    tests: list[tuple[str, dict[str, float]]] = []
    present = set(spec.param_names)
    frozen = set(spec.frozen_indices)
    p = dict(zip(spec.param_names, best_full))

    for idx, name in enumerate(spec.param_names):
        if idx in frozen:
            continue
        if name in {"lambda1", "lambda2", "lambda3", "lambda4", "c0", "c1"}:
            tests.append((f"{name}_zero", {name: 0.0}))
        elif name == "alpha":
            tests.append(("alpha_one", {"alpha": 1.0}))

    if {"amp", "logx0", "sigx"}.issubset(present) and spec.param_names.index("amp") not in frozen:
        tests.append(("bump_off", {"amp": 0.0, "logx0": p["logx0"], "sigx": p["sigx"]}))

    deduped: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for name, fmap in tests:
        key = json.dumps([name, fmap], sort_keys=True)
        if key not in seen:
            deduped.append((name, fmap))
            seen.add(key)
    return deduped


def max_abs_corr(corr: np.ndarray) -> float:
    if corr.size == 0:
        return float("nan")
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    return float(np.max(vals)) if vals.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fit-name", required=True)
    parser.add_argument("--card-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--nested-maxfun", type=int, default=120)
    parser.add_argument("--rel-step", type=float, default=2e-4)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    card_path = Path(args.card_path)
    result_path = Path(args.result_path)
    best_full = load_best_full(result_path)
    temp_fit_name = f"{args.fit_name}__SignificanceTmp"
    temp_card = write_temp_card(card_path, temp_fit_name, best_full)

    spec = make_spec(args.name, temp_fit_name, temp_card, result_path)
    session = FitSession(spec)
    n_total = int(sum(session.n_list.values()))
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0, rel_step=args.rel_step)
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10**2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    _, cov_norm = psd_covariance(H_total)
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    free_param_names = session.free_param_names
    free_name_to_idx = {name: i for i, name in enumerate(free_param_names)}
    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    sigma_norm = np.sqrt(np.maximum(np.diag(cov_norm), 0.0))

    param_rows = []
    best_full_arr = np.asarray(best_eval["full_params"], dtype=float)
    for name, value in zip(spec.param_names, best_full_arr):
        ref = REFERENCE_MAP.get(name, np.nan)
        row: dict[str, float | str | bool] = {
            "model": args.name,
            "label": args.label or args.name,
            "param": name,
            "value": float(value),
            "reference": float(ref) if not pd.isna(ref) else np.nan,
            "is_free": name in free_name_to_idx,
        }
        if name in free_name_to_idx:
            idx = free_name_to_idx[name]
            sig_n = float(sigma_norm[idx])
            sig_p = float(sigma_phys[idx])
            row["sigma_norm"] = sig_n
            row["sigma_phys"] = sig_p
            if not pd.isna(ref) and sig_p > 0:
                z = (value - ref) / sig_p
                row["z_vs_reference"] = float(z)
                row["signal_to_sigma"] = float(abs(z))
            else:
                row["z_vs_reference"] = np.nan
                row["signal_to_sigma"] = np.nan
            row["frac_uncertainty"] = float(sig_p / abs(value)) if abs(value) > 1e-12 else np.nan
        else:
            row["sigma_norm"] = np.nan
            row["sigma_phys"] = np.nan
            row["z_vs_reference"] = np.nan
            row["signal_to_sigma"] = np.nan
            row["frac_uncertainty"] = np.nan
        param_rows.append(row)

    corr_rows = top_correlations(corr, free_param_names, top_n=16)

    nested_rows = []
    for test_name, fixed_map in null_tests_for(best_full_arr.tolist(), spec):
        nested = NestedFit(session, best_full_arr, fixed_map)
        fit_info = nested.solve(maxfun=args.nested_maxfun)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        metrics = best_nested["metrics"]
        nested_rows.append(
            {
                "model": args.name,
                "label": args.label or args.name,
                "test": test_name,
                "fixed_map": json.dumps(fixed_map, sort_keys=True),
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2_total": metrics["chi2dN_total"] * n_total,
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "delta_chi2dN": metrics["chi2dN_total"] - chi2dN,
                "delta_chi2_total": (metrics["chi2dN_total"] - chi2dN) * n_total,
                "delta_absdev": metrics["highE_mean_absdev_first3"] - best_eval["metrics"]["highE_mean_absdev_first3"],
                "delta_shortfall": metrics["highE_mean_shortfall_first3"] - best_eval["metrics"]["highE_mean_shortfall_first3"],
                "fit_evals": fit_info["nf"],
                "fit_elapsed_s": fit_info["elapsed_s"],
            }
        )

    eigvals = np.linalg.eigvalsh(0.5 * (H_total + H_total.T))
    top_corr_pair = corr_rows[0] if corr_rows else {"param_i": "", "param_j": "", "corr": np.nan}
    summary = {
        "model": args.name,
        "label": args.label or args.name,
        "fit_name": args.fit_name,
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_evals": h["nevals"],
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(eigvals)),
        "hessian_max_eig": float(np.max(eigvals)),
        "max_abs_corr": max_abs_corr(corr),
        "top_corr_param_i": top_corr_pair.get("param_i", ""),
        "top_corr_param_j": top_corr_pair.get("param_j", ""),
        "top_corr": float(top_corr_pair.get("corr", np.nan)),
    }

    pd.DataFrame(param_rows).to_csv(results_dir / "parameter_uncertainty.csv", index=False)
    pd.DataFrame(corr, index=free_param_names, columns=free_param_names).to_csv(results_dir / "correlation_matrix.csv")
    pd.DataFrame(corr_rows).to_csv(results_dir / "top_correlations.csv", index=False)
    nested_df = pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True])
    nested_df.to_csv(results_dir / "nested_tests.csv", index=False)
    nested_df.head(10).to_csv(results_dir / "nested_top.csv", index=False)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
