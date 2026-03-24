from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CandidateSpec, FITS_DIR, FitSession
from parameter_significance_0_2 import NestedFit, finite_diff_hessian, parse_array, parse_struct_fields, top_correlations


ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "Cards"
RESULTS_DIR = FITS_DIR / "broad_bump_42_profile_results"
RESULTS_DIR.mkdir(exist_ok=True)

CARD_PATH = CARDS_DIR / "BroadBump42LogGaussAlpha1NoLambda2.jl"
RESULT_PATH = FITS_DIR / "broad_bump_42_prune_results" / "broad_bump_42_alpha1_lambda2_zero.json"


def load_best_full(result_path: Path) -> list[float]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


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


def make_spec() -> CandidateSpec:
    best_full = load_best_full(RESULT_PATH)
    temp_fit_name = "BroadBump42LogGaussAlpha1NoLambda2__ProfileTmp"
    temp_card = write_temp_card(CARD_PATH, temp_fit_name, best_full)
    card_text = temp_card.read_text(encoding="utf-8")
    return CandidateSpec(
        name="broad_bump_42_alpha1_lambda2_zero_profile",
        fit_name=temp_fit_name,
        np_name=card_text.split('const NP_name = "', 1)[1].split('"', 1)[0],
        param_names=parse_struct_fields(card_text),
        initial_params=best_full,
        bounds=[(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="profile_scan",
    )


def regularized_corr(H_total: np.ndarray, names: list[str], rel_cut: float) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    Hs = 0.5 * (H_total + H_total.T)
    evals, evecs = np.linalg.eigh(Hs)
    max_pos = float(np.max(evals))
    keep = evals > rel_cut * max_pos
    if not np.any(keep):
        keep[np.argmax(evals)] = True
    inv = np.zeros_like(evals)
    inv[keep] = 1.0 / evals[keep]
    cov = (evecs * inv) @ evecs.T
    diag = np.maximum(np.diag(cov), 1e-300)
    corr = cov / np.sqrt(np.outer(diag, diag))
    rows = top_correlations(corr, names, top_n=12)
    summary = {
        "rel_cut": rel_cut,
        "n_keep": int(np.sum(keep)),
        "max_abs_corr": float(np.max(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)]))),
        "top_param_i": rows[0]["param_i"] if rows else "",
        "top_param_j": rows[0]["param_j"] if rows else "",
        "top_corr": float(rows[0]["corr"]) if rows else float("nan"),
    }
    return corr, rows, summary


def profile_pair(session: FitSession, best_full: np.ndarray, pair: tuple[str, str], grid_n: int = 5, maxfun: int = 60) -> pd.DataFrame:
    name_to_idx = {name: i for i, name in enumerate(session.spec.param_names)}
    idx_i = name_to_idx[pair[0]]
    idx_j = name_to_idx[pair[1]]

    theta0 = session.normalize_params(best_full[session.free_idx])
    free_name_to_pos = {name: i for i, name in enumerate(session.free_param_names)}
    pos_i = free_name_to_pos[pair[0]]
    pos_j = free_name_to_pos[pair[1]]

    norm_i = float(theta0[pos_i])
    norm_j = float(theta0[pos_j])
    span = 0.08
    grid_i = np.linspace(max(0.0, norm_i - span), min(1.0, norm_i + span), grid_n)
    grid_j = np.linspace(max(0.0, norm_j - span), min(1.0, norm_j + span), grid_n)

    rows = []
    total = grid_n * grid_n
    done = 0
    for ni in grid_i:
        vi = float(session.lower_bounds[pos_i] + ni * (session.upper_bounds[pos_i] - session.lower_bounds[pos_i]))
        for nj in grid_j:
            vj = float(session.lower_bounds[pos_j] + nj * (session.upper_bounds[pos_j] - session.lower_bounds[pos_j]))
            fixed = {pair[0]: vi, pair[1]: vj}
            nested = NestedFit(session, best_full, fixed)
            fit_info = nested.solve(maxfun=maxfun)
            best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
            metrics = best_nested["metrics"]
            rows.append(
                {
                    "param_i": pair[0],
                    "param_j": pair[1],
                    "value_i": vi,
                    "value_j": vj,
                    "chi2dN_total": metrics["chi2dN_total"],
                    "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                    "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                    "fit_evals": fit_info["nf"],
                    "fit_elapsed_s": fit_info["elapsed_s"],
                }
            )
            done += 1
            print(f"[profile {pair[0]}/{pair[1]}] {done}/{total}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    spec = make_spec()
    session = FitSession(spec)
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    best_full = np.asarray(best_eval["full_params"], dtype=float)

    print("[hessian] finite differences", flush=True)
    h = finite_diff_hessian(session.objective_log_normalized, session.theta0, rel_step=2e-4)
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    n_total = int(sum(session.n_list.values()))
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10 ** 2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN

    free_names = session.free_param_names
    cutoff_summaries = []
    for rel_cut in [1e-2, 3e-3, 1e-3]:
        corr, rows, summary = regularized_corr(H_total, free_names, rel_cut)
        tag = f"cut_{rel_cut:.0e}".replace("-", "m")
        pd.DataFrame(corr, index=free_names, columns=free_names).to_csv(RESULTS_DIR / f"corr_{tag}.csv")
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"top_corr_{tag}.csv", index=False)
        cutoff_summaries.append(summary)

    pd.DataFrame(cutoff_summaries).to_csv(RESULTS_DIR / "regularized_corr_summary.csv", index=False)

    for pair in [("lambda1", "lambda3"), ("amp", "BNP")]:
        print(f"[scan] {pair[0]} vs {pair[1]}", flush=True)
        df = profile_pair(session, best_full, pair, grid_n=5, maxfun=60)
        df.to_csv(RESULTS_DIR / f"profile_{pair[0]}_{pair[1]}.csv", index=False)

    summary = {
        "chi2dN_total": chi2dN,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_min_eig": float(np.min(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
        "hessian_max_eig": float(np.max(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
