from __future__ import annotations

import json
import math
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
RESULTS_DIR = FITS_DIR / "parameter_significance_xbar_quad_no_lambda4_results"
RESULTS_DIR.mkdir(exist_ok=True)

CARD_PATH = CARDS_DIR / "BroadBump42XbarQuadLogPairNoLambda4.jl"
RESULT_PATH = FITS_DIR / "log_basis_42_nested_prune_results" / "xbar_quad_logpair_bump_lambda4_zero_nested.json"


def load_best_full(result_path: Path) -> list[float]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def make_spec() -> CandidateSpec:
    card_text = CARD_PATH.read_text(encoding="utf-8")
    return CandidateSpec(
        name="xbar_quad_logpair_bump_lambda4_zero",
        fit_name="BroadBump42XbarQuadLogPairNoLambda4",
        np_name=card_text.split('const NP_name = "', 1)[1].split('"', 1)[0],
        param_names=parse_struct_fields(card_text),
        initial_params=load_best_full(RESULT_PATH),
        bounds=[(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="significance",
    )


def main() -> None:
    spec = make_spec()
    session = FitSession(spec)
    n_total = int(sum(session.n_list.values()))
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0, rel_step=2e-4)
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10**2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    _, cov_norm = psd_covariance(H_total)
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    free_param_names = session.free_param_names
    corr_rows = top_correlations(corr, free_param_names, top_n=16)

    best_full = np.asarray(best_eval["full_params"], dtype=float)
    nested_rows = []
    for test_name, fixed_map in [
        ("lambda1_zero", {"lambda1": 0.0}),
        ("lambda2_zero", {"lambda2": 0.0}),
        ("lambda3_zero", {"lambda3": 0.0}),
        ("c0_zero", {"c0": 0.0}),
        ("c1_zero", {"c1": 0.0}),
        ("bump_off", {"amp": 0.0, "logx0": float(best_full[5]), "sigx": float(best_full[6])}),
    ]:
        nested = NestedFit(session, best_full, fixed_map)
        fit_info = nested.solve(maxfun=120)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        metrics = best_nested["metrics"]
        nested_rows.append(
            {
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
    max_corr = float(np.max(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])))
    summary = {
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_evals": h["nevals"],
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(eigvals)),
        "hessian_max_eig": float(np.max(eigvals)),
        "max_abs_corr": max_corr,
        "top_corr_param_i": corr_rows[0]["param_i"] if corr_rows else "",
        "top_corr_param_j": corr_rows[0]["param_j"] if corr_rows else "",
        "top_corr": float(corr_rows[0]["corr"]) if corr_rows else float("nan"),
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(corr, index=free_param_names, columns=free_param_names).to_csv(RESULTS_DIR / "correlation_matrix.csv")
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "top_correlations.csv", index=False)
    pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).to_csv(
        RESULTS_DIR / "nested_tests.csv", index=False
    )


if __name__ == "__main__":
    main()
