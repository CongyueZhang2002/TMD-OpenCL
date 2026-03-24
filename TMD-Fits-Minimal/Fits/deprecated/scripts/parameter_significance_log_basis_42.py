from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import FITS_DIR, CandidateSpec, FitSession
from parameter_significance_0_2 import (
    NestedFit,
    finite_diff_hessian,
    parse_array,
    parse_struct_fields,
    psd_covariance,
    top_correlations,
)


ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "Cards"
RESULTS_DIR = FITS_DIR / "parameter_significance_log_basis_42_results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    label: str
    fit_name: str
    card_path: Path
    result_path: Path


MODELS = [
    ModelConfig(
        name="broad_bump_42_alpha1_lambda2_zero",
        label="Broad bump 4-2, no lambda2",
        fit_name="BroadBump42LogGaussAlpha1NoLambda2",
        card_path=CARDS_DIR / "BroadBump42LogGaussAlpha1NoLambda2.jl",
        result_path=FITS_DIR / "broad_bump_42_prune_results" / "broad_bump_42_alpha1_lambda2_zero.json",
    ),
    ModelConfig(
        name="xbar_quad_logpair_bump",
        label="xbar + quad + logx + log1mx + bump",
        fit_name="BroadBump42XbarQuadLogPair",
        card_path=CARDS_DIR / "BroadBump42XbarQuadLogPair.jl",
        result_path=FITS_DIR / "log_basis_42_results" / "xbar_quad_logpair_bump.json",
    ),
    ModelConfig(
        name="const_quad_logpair_bump",
        label="const + quad + logx + log1mx + bump",
        fit_name="BroadBump42ConstQuadLogPair",
        card_path=CARDS_DIR / "BroadBump42ConstQuadLogPair.jl",
        result_path=FITS_DIR / "log_basis_42_results" / "const_quad_logpair_bump.json",
    ),
    ModelConfig(
        name="const_logpair_bump",
        label="const + logx + log1mx + bump",
        fit_name="BroadBump42ConstLogPair",
        card_path=CARDS_DIR / "BroadBump42ConstLogPair.jl",
        result_path=FITS_DIR / "log_basis_42_results" / "const_logpair_bump.json",
    ),
]


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


def make_spec(model: ModelConfig) -> CandidateSpec:
    card_text = model.card_path.read_text(encoding="utf-8")
    param_names = parse_struct_fields(card_text)
    best_full = load_best_full(model.result_path)
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]
    np_name = card_text.split('const NP_name = "', 1)[1].split('"', 1)[0]

    return CandidateSpec(
        name=model.name,
        fit_name=model.fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=best_full,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="log_basis_42_significance",
    )


def null_tests_for(best_full: list[float], spec: CandidateSpec) -> list[tuple[str, dict[str, float]]]:
    tests: list[tuple[str, dict[str, float]]] = []
    present = set(spec.param_names)
    if "lambda1" in present:
        tests.append(("lambda1_zero", {"lambda1": 0.0}))
    if "lambda2" in present and 1 not in spec.frozen_indices:
        tests.append(("lambda2_zero", {"lambda2": 0.0}))
    if "lambda3" in present and spec.param_names.count("lambda3") and spec.param_names.index("lambda3") not in spec.frozen_indices:
        tests.append(("lambda3_zero", {"lambda3": 0.0}))
    if "lambda4" in present and spec.param_names.index("lambda4") not in spec.frozen_indices:
        tests.append(("lambda4_zero", {"lambda4": 0.0}))
    if "c0" in present:
        tests.append(("c0_zero", {"c0": 0.0}))
    if "c1" in present:
        tests.append(("c1_zero", {"c1": 0.0}))
    if "amp" in present:
        p = dict(zip(spec.param_names, best_full))
        tests.append(("bump_off", {"amp": 0.0, "logx0": p["logx0"], "sigx": p["sigx"]}))
    return tests


def max_abs_corr(corr: np.ndarray) -> float:
    if corr.size == 0:
        return float("nan")
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    return float(np.max(vals)) if vals.size else float("nan")


def analyze_model(model: ModelConfig, nested_maxfun: int, rel_step: float) -> dict[str, float]:
    spec = make_spec(model)
    session = FitSession(spec)
    n_total = int(sum(session.n_list.values()))
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    best_full = np.asarray(best_eval["full_params"], dtype=float)

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0, rel_step=rel_step)
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10**2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    H_psd, cov_norm = psd_covariance(H_total)

    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    sigma_norm = np.sqrt(np.maximum(np.diag(cov_norm), 0.0))
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    free_param_names = session.free_param_names
    free_name_to_idx = {name: i for i, name in enumerate(free_param_names)}

    param_rows = []
    for name, value in zip(spec.param_names, best_full):
        ref = REFERENCE_MAP.get(name, np.nan)
        is_free = name in free_name_to_idx
        row: dict[str, float | str | bool] = {
            "model": model.name,
            "param": name,
            "is_free": is_free,
            "value": float(value),
            "reference": float(ref) if not pd.isna(ref) else np.nan,
        }
        if is_free:
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
    for test_name, fixed_map in null_tests_for(best_full.tolist(), spec):
        nested = NestedFit(session, best_full, fixed_map)
        fit_info = nested.solve(maxfun=nested_maxfun)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        metrics = best_nested["metrics"]
        nested_rows.append(
            {
                "model": model.name,
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
    max_corr = max_abs_corr(corr)
    top_corr_pair = corr_rows[0] if corr_rows else {"param_i": "", "param_j": "", "corr": np.nan}
    summary = {
        "model": model.name,
        "label": model.label,
        "fit_name": model.fit_name,
        "n_total": n_total,
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_evals": h["nevals"],
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(eigvals)),
        "hessian_max_eig": float(np.max(eigvals)),
        "max_abs_corr": max_corr,
        "top_corr_param_i": top_corr_pair.get("param_i", ""),
        "top_corr_param_j": top_corr_pair.get("param_j", ""),
        "top_corr": float(top_corr_pair.get("corr", np.nan)),
    }

    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / f"{model.name}_parameter_uncertainty.csv", index=False)
    pd.DataFrame(corr, index=free_param_names, columns=free_param_names).to_csv(
        RESULTS_DIR / f"{model.name}_correlation_matrix.csv"
    )
    pd.DataFrame(H_total, index=free_param_names, columns=free_param_names).to_csv(
        RESULTS_DIR / f"{model.name}_hessian_total.csv"
    )
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / f"{model.name}_top_correlations.csv", index=False)
    nested_df = pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True])
    nested_df.to_csv(RESULTS_DIR / f"{model.name}_nested_tests.csv", index=False)
    strongest = nested_df.head(8).copy()
    strongest.insert(0, "rank", range(1, len(strongest) + 1))
    strongest.to_csv(RESULTS_DIR / f"{model.name}_nested_top.csv", index=False)
    (RESULTS_DIR / f"{model.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-maxfun", type=int, default=120)
    parser.add_argument("--rel-step", type=float, default=2e-4)
    args = parser.parse_args()

    summaries = [analyze_model(model, nested_maxfun=args.nested_maxfun, rel_step=args.rel_step) for model in MODELS]
    summary_df = pd.DataFrame(summaries).sort_values(
        ["chi2dN_total", "max_abs_corr", "highE_mean_absdev_first3"]
    ).reset_index(drop=True)
    summary_df.to_csv(RESULTS_DIR / "model_summaries.csv", index=False)

    lines = [
        "# Parameter Significance And Correlation For Log-Basis 4-2 Models",
        "",
        "Models analyzed:",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['model']}`: chi2/N `{row['chi2dN_total']:.6f}`, "
            f"absdev `{row['highE_mean_absdev_first3']:.6f}`, "
            f"shortfall `{row['highE_mean_shortfall_first3']:.6f}`, "
            f"max|corr| `{row['max_abs_corr']:.3f}`"
        )
    lines.append("")
    lines.append("Saved files:")
    lines.append("")
    lines.append("- `model_summaries.csv`")
    lines.append("- `<model>_parameter_uncertainty.csv`")
    lines.append("- `<model>_correlation_matrix.csv`")
    lines.append("- `<model>_top_correlations.csv`")
    lines.append("- `<model>_nested_tests.csv`")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
