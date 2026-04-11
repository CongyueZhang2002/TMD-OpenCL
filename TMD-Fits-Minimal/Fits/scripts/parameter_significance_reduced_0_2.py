from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CandidateSpec, FitSession
from parameter_significance_0_2 import NestedFit, finite_diff_hessian, psd_covariance, top_correlations
from scan_table_variants import parse_array, parse_struct_fields
from _paths import CARDS_DIR, FITS_DIR, RESULTS_ROOT, ROOT


RESULTS_DIR = RESULTS_ROOT / "parameter_significance_reduced_0_2_results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ReducedModel:
    name: str
    label: str
    fit_name: str
    np_name: str
    card_path: Path
    result_path: Path


MODELS = [
    ReducedModel(
        name="poly_bstar_cslog_alpha1",
        label="Poly-x bstar + CSlog alpha=1",
        fit_name="Art23FamilyMuPolyBstarCSLogAlpha1",
        np_name="NP-Art23FamilyMuPolyBstarCSLogAlpha1.cl",
        card_path=CARDS_DIR / "Art23FamilyMuPolyBstarCSLogAlpha1.jl",
        result_path=RESULTS_ROOT / "pruned_model_refits_0_2_results" / "poly_bstar_cslog_alpha1.json",
    ),
    ReducedModel(
        name="poly_bstar_cslog_loggauss_reduced",
        label="Poly-x bstar + CSlog + loggauss reduced",
        fit_name="PolyBstarCSLogLogGaussReduced",
        np_name="NP-PolyBstarCSLogLogGaussReduced.cl",
        card_path=CARDS_DIR / "PolyBstarCSLogLogGaussReduced.jl",
        result_path=RESULTS_ROOT / "pruned_model_refits_0_2_results" / "poly_bstar_cslog_loggauss_lambda3_0_alpha1.json",
    ),
]


def base_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["best"]["metrics"]


def make_spec(model: ReducedModel) -> CandidateSpec:
    card_text = model.card_path.read_text(encoding="utf-8")
    param_names = parse_struct_fields(card_text)
    initial = [float(x) for x in parse_array(card_text, "initial_params")]
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]
    return CandidateSpec(
        name=model.name,
        fit_name=model.fit_name,
        np_name=model.np_name,
        param_names=param_names,
        initial_params=initial,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="reduced_significance",
    )


def null_tests(model_name: str, best_full: list[float], param_names: list[str]) -> list[tuple[str, dict[str, float]]]:
    p = dict(zip(param_names, best_full))
    if model_name == "poly_bstar_cslog_alpha1":
        return [
            ("lambda3_zero", {"lambda3": 0.0}),
            ("lambda4_zero", {"lambda4": 0.0}),
            ("c1_zero", {"c1": 0.0}),
        ]
    if model_name == "poly_bstar_cslog_loggauss_reduced":
        return [
            ("c1_zero", {"c1": 0.0}),
            ("lambda4_zero", {"lambda4": 0.0}),
            ("bump_off", {"amp": 0.0, "logx0": p["logx0"], "sigx": p["sigx"]}),
        ]
    raise ValueError(model_name)


def analyze_model(model: ReducedModel, nested_maxfun: int) -> None:
    spec = make_spec(model)
    session = FitSession(spec)
    n_total = int(sum(session.n_list.values()))
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    best_full = np.asarray(best_eval["full_params"], dtype=float)

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0)
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total
    ln10 = np.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10 ** 2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    H_psd, cov_norm = psd_covariance(H_total)

    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    sigma_norm = np.sqrt(np.maximum(np.diag(cov_norm), 0.0))
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    param_rows = []
    for name, value, sig_n, sig_p in zip(spec.param_names, best_full, sigma_norm, sigma_phys):
        row = {
            "model": model.name,
            "param": name,
            "value": float(value),
            "sigma_norm": float(sig_n),
            "sigma_phys": float(sig_p),
        }
        if abs(value) > 1e-12:
            row["frac_uncertainty"] = float(sig_p / abs(value))
        else:
            row["frac_uncertainty"] = np.nan
        param_rows.append(row)

    corr_rows = top_correlations(corr, spec.param_names, top_n=12)

    nested_rows = []
    for test_name, fixed_map in null_tests(model.name, best_full.tolist(), spec.param_names):
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

    base_summary = {
        "model": model.name,
        "label": model.label,
        "n_total": n_total,
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_evals": h["nevals"],
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
        "hessian_max_eig": float(np.max(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
    }

    (RESULTS_DIR / f"{model.name}_summary.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / f"{model.name}_parameter_uncertainty.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / f"{model.name}_top_correlations.csv", index=False)
    pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).to_csv(
        RESULTS_DIR / f"{model.name}_nested_tests.csv", index=False
    )


def summarize() -> None:
    summary_rows = []
    nested_best_rows = []
    for model in MODELS:
        path = RESULTS_DIR / f"{model.name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary_rows.append(data)

        nested_path = RESULTS_DIR / f"{model.name}_nested_tests.csv"
        if nested_path.exists():
            df_nested = pd.read_csv(nested_path)
            strongest = df_nested.sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).head(5).copy()
            strongest.insert(0, "rank", range(1, len(strongest) + 1))
            strongest.to_csv(RESULTS_DIR / f"{model.name}_nested_top5.csv", index=False)
            for _, row in strongest.iterrows():
                nested_best_rows.append(
                    {
                        "model": model.name,
                        "rank": int(row["rank"]),
                        "test": row["test"],
                        "delta_chi2_total": row["delta_chi2_total"],
                        "delta_absdev": row["delta_absdev"],
                        "delta_shortfall": row["delta_shortfall"],
                    }
                )

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("chi2dN_total").to_csv(RESULTS_DIR / "model_summaries.csv", index=False)
    if nested_best_rows:
        pd.DataFrame(nested_best_rows).to_csv(RESULTS_DIR / "nested_top5_all_models.csv", index=False)

    lines = [
        "# Reduced Model Significance On Default 0-2",
        "",
        "Method:",
        "",
        "- Local finite-difference Hessian around the reduced-model 0-2 fit.",
        "- Nested local refits with one remaining optional term turned off.",
        "",
        "Analyzed reduced models:",
        "",
    ]
    for row in sorted(summary_rows, key=lambda item: item["chi2dN_total"]):
        lines.append(
            f"- `{row['model']}`: chi2/N `{row['chi2dN_total']:.6f}`, "
            f"highE absdev `{row['highE_mean_absdev_first3']:.6f}`, "
            f"highE shortfall `{row['highE_mean_shortfall_first3']:.6f}`"
        )
    lines.extend(["", "See per-model CSV files for parameter uncertainties, top correlations, and nested tests.", ""])
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def orchestrate(nested_maxfun: int) -> None:
    for model in MODELS:
        cmd = [
            str(Path(sys.executable)),
            str(Path(__file__)),
            "--model",
            model.name,
            "--nested-maxfun",
            str(nested_maxfun),
        ]
        print(f"\n=== Reduced significance {model.name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"exit code {proc.returncode}")
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--nested-maxfun", type=int, default=60)
    args = parser.parse_args()

    if args.model:
        model = next(item for item in MODELS if item.name == args.model)
        analyze_model(model, nested_maxfun=args.nested_maxfun)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(nested_maxfun=args.nested_maxfun)


if __name__ == "__main__":
    main()
