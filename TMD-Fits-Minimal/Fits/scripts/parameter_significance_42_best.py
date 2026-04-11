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
from _paths import RESULTS_ROOT


ROOT = FITS_DIR.parent
CARDS_DIR = ROOT / "Cards"
RESULTS_DIR = RESULTS_ROOT / "parameter_significance_42_best_results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    label: str
    fit_name: str
    card_path: Path
    result_path: Path


MODEL = ModelConfig(
    name="poly_bstar_cslog_42best",
    label="Poly-x bstar + CSlog (4-2 best, no bump)",
    fit_name="Art23FamilyMuPolyBstarCSLog42Best",
    card_path=CARDS_DIR / "Art23FamilyMuPolyBstarCSLog42Best.jl",
    result_path=RESULTS_ROOT / "aggressive_table_refit_results" / "poly_bstar_cslog__4-2.json",
)


REFERENCE_MAP = {
    "lambda1": 0.0,
    "lambda2": 0.0,
    "lambda3": 0.0,
    "lambda4": 0.0,
    "alpha": 1.0,
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
        kernel_variant="4_2_significance",
    )


def null_tests_for(best_full: list[float], param_names: list[str]) -> list[tuple[str, dict[str, float]]]:
    return [
        ("lambda1_zero", {"lambda1": 0.0}),
        ("lambda2_zero", {"lambda2": 0.0}),
        ("lambda3_zero", {"lambda3": 0.0}),
        ("lambda4_zero", {"lambda4": 0.0}),
        ("alpha_is_1", {"alpha": 1.0}),
        ("c0_zero", {"c0": 0.0}),
        ("c1_zero", {"c1": 0.0}),
    ]


def analyze_model(model: ModelConfig, nested_maxfun: int, rel_step: float) -> None:
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

    param_rows = []
    for name, value, sig_n, sig_p in zip(spec.param_names, best_full, sigma_norm, sigma_phys):
        ref = REFERENCE_MAP.get(name, np.nan)
        row: dict[str, float | str] = {
            "model": model.name,
            "param": name,
            "value": float(value),
            "sigma_norm": float(sig_n),
            "sigma_phys": float(sig_p),
            "reference": float(ref) if not pd.isna(ref) else np.nan,
        }
        if not pd.isna(ref) and sig_p > 0:
            z = (value - ref) / sig_p
            row["z_vs_reference"] = float(z)
            row["signal_to_sigma"] = float(abs(z))
        else:
            row["z_vs_reference"] = np.nan
            row["signal_to_sigma"] = np.nan
        if abs(value) > 1e-12:
            row["frac_uncertainty"] = float(sig_p / abs(value))
        else:
            row["frac_uncertainty"] = np.nan
        param_rows.append(row)

    corr_rows = top_correlations(corr, spec.param_names, top_n=14)

    nested_rows = []
    for test_name, fixed_map in null_tests_for(best_full.tolist(), spec.param_names):
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
    }

    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / f"{model.name}_parameter_uncertainty.csv", index=False)
    pd.DataFrame(corr, index=spec.param_names, columns=spec.param_names).to_csv(RESULTS_DIR / f"{model.name}_correlation_matrix.csv")
    pd.DataFrame(H_total, index=spec.param_names, columns=spec.param_names).to_csv(RESULTS_DIR / f"{model.name}_hessian_total.csv")
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / f"{model.name}_top_correlations.csv", index=False)
    pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).to_csv(
        RESULTS_DIR / f"{model.name}_nested_tests.csv", index=False
    )
    strongest = pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).head(7).copy()
    strongest.insert(0, "rank", range(1, len(strongest) + 1))
    strongest.to_csv(RESULTS_DIR / f"{model.name}_nested_top.csv", index=False)
    (RESULTS_DIR / f"{model.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Parameter Significance For 4-2 Best No-Bump Model",
        "",
        f"- Model: `{model.label}`",
        f"- Fit name: `{model.fit_name}`",
        f"- chi2/N: `{chi2dN:.6f}`",
        f"- total chi2: `{chi2_total:.3f}`",
        f"- highE absdev: `{best_eval['metrics']['highE_mean_absdev_first3']:.6f}`",
        f"- highE shortfall: `{best_eval['metrics']['highE_mean_shortfall_first3']:.6f}`",
        "",
        "Method:",
        "",
        "- Local finite-difference Hessian around the saved aggressive-refit 4-2 point.",
        "- PSD-projected covariance from the chi2 Hessian.",
        "- Nested local refits with one term set to its natural null/reference value.",
        "",
        "Saved files:",
        "",
        f"- `{model.name}_parameter_uncertainty.csv`",
        f"- `{model.name}_correlation_matrix.csv`",
        f"- `{model.name}_top_correlations.csv`",
        f"- `{model.name}_hessian_total.csv`",
        f"- `{model.name}_nested_tests.csv`",
        "",
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-maxfun", type=int, default=120)
    parser.add_argument("--rel-step", type=float, default=2e-4)
    args = parser.parse_args()
    analyze_model(MODEL, nested_maxfun=args.nested_maxfun, rel_step=args.rel_step)


if __name__ == "__main__":
    main()
