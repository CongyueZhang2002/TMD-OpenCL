from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from art23_family_search import Art23FamilyFitSession
from auto_np_search import CandidateSpec, FITS_DIR
from _paths import RESULTS_ROOT


RESULTS_DIR = RESULTS_ROOT / "super_refit_broad_bump_42_results"
RESULTS_DIR.mkdir(exist_ok=True)

PARAM_NAMES = [
    "lambda1",  # log(x)
    "lambda2",  # (1-x)
    "lambda3",  # x(1-x)
    "logx0",
    "sigx",
    "amp",
    "BNP",
    "c0",
    "c1",
]


CARD_PARAMS = [
    0.04375743605,
    0.9774184714,
    -1.960659735,
    -4.99216975,
    0.717326296,
    -0.338498232,
    1.455523349,
    0.0717885484,
    0.02583083271,
]

PREVIOUS_DEEP_PARAMS = [
    0.02359248944481407,
    1.0547935947065572,
    -2.354456696325701,
    -5.211365058158037,
    1.1037690127446673,
    -0.4324625556574935,
    1.493301919733006,
    0.07001502121630351,
    0.027699782530320217,
]


CURRENT_SPEC = CandidateSpec(
    name="broad_bump_42_alpha1_nolambda2_super",
    fit_name="BroadBump42LogGaussAlpha1NoLambda2",
    np_name="NP-BroadBump42LogGaussAlpha1NoLambda2.cl",
    param_names=PARAM_NAMES,
    initial_params=PREVIOUS_DEEP_PARAMS,
    bounds=[
        (-0.5, 0.5),
        (0.02, 8.0),
        (-10.0, 10.0),
        (-9.210340372, -1.203972804),
        (0.6, 2.5),
        (-3.0, 3.0),
        (0.4, 4.5),
        (0.0, 0.25),
        (0.0, 0.25),
    ],
    frozen_indices=[],
    kernel_variant="loggauss",
)


class SuperBudgetFitSession(Art23FamilyFitSession):
    def __init__(self, spec: CandidateSpec) -> None:
        super().__init__(spec)
        self.card_theta = self.normalize_params(np.asarray(CARD_PARAMS, dtype=float))
        self.prev_theta = self.normalize_params(np.asarray(PREVIOUS_DEEP_PARAMS, dtype=float))

    def candidate_start_points(self) -> list[np.ndarray]:
        starts = [self.theta0.copy(), self.card_theta.copy(), self.prev_theta.copy()]
        seed = 20260324 + 37 * sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for base in [self.theta0, self.card_theta, self.prev_theta]:
            for scale in [0.015, 0.03, 0.05, 0.08, 0.12, 0.18]:
                for _ in range(2):
                    starts.append(np.clip(base + rng.normal(0.0, scale, size=base.shape), 0.0, 1.0))

        anchor_values = {
            "lambda1": [0.05, 0.20, 0.35, 0.50, 0.70, 0.90],
            "lambda2": [0.05, 0.20, 0.35, 0.50, 0.70, 0.90],
            "lambda3": [0.05, 0.20, 0.35, 0.50, 0.70, 0.90],
            "logx0": [0.18, 0.30, 0.42, 0.55, 0.68, 0.80],
            "sigx": [0.05, 0.16, 0.28, 0.42, 0.58, 0.78],
            "amp": [0.12, 0.28, 0.45, 0.62, 0.82],
            "BNP": [0.08, 0.22, 0.38, 0.55, 0.72, 0.90],
            "c0": [0.05, 0.18, 0.34, 0.52, 0.72, 0.90],
            "c1": [0.05, 0.18, 0.34, 0.52, 0.72, 0.90],
        }
        for pname, vals in anchor_values.items():
            idx = self.free_param_names.index(pname)
            for val in vals:
                start = self.theta0.copy()
                start[idx] = val
                starts.append(start)

        for _ in range(18):
            starts.append(np.clip(rng.uniform(0.0, 1.0, size=self.theta0.shape), 0.0, 1.0))

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique[:42]

    def fit(self, stage2_maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1_budget = max(120, stage2_maxfun // 5)
        refine_budget = max(320, stage2_maxfun // 2)
        polish_budget = max(220, stage2_maxfun // 3)
        verify_budget = max(140, stage2_maxfun // 5)

        stage1_results = []
        starts = self.candidate_start_points()
        print(f"[super-refit] stage1 starts={len(starts)} budget={stage1_budget}", flush=True)
        for i, start in enumerate(starts):
            print(f"[super-refit] stage1 {i + 1}/{len(starts)}", flush=True)
            res = self._solve(start, maxfun=stage1_budget, rhobeg=0.18, rhoend=8e-4, seek_global_minimum=True)
            stage1_results.append(
                {
                    "start_id": i,
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
        stage1_results.sort(key=lambda item: item["log10_chi2"])
        print(f"[super-refit] stage1 best={stage1_results[0]['log10_chi2']:.8f}", flush=True)

        stage2_results = []
        stage2_pool = stage1_results[:8]
        print(f"[super-refit] stage2 pool={len(stage2_pool)} budget={stage2_maxfun}", flush=True)
        for i, item in enumerate(stage2_pool):
            print(f"[super-refit] stage2 {i + 1}/{len(stage2_pool)}", flush=True)
            res = self._solve(item["theta"], maxfun=stage2_maxfun, rhobeg=0.10, rhoend=8e-5, seek_global_minimum=True)
            stage2_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
        stage2_results.sort(key=lambda item: item["log10_chi2"])
        print(f"[super-refit] stage2 best={stage2_results[0]['log10_chi2']:.8f}", flush=True)

        refine_results = []
        refine_pool = stage2_results[:4]
        print(f"[super-refit] refine pool={len(refine_pool)} budget={refine_budget}", flush=True)
        for i, item in enumerate(refine_pool):
            print(f"[super-refit] refine {i + 1}/{len(refine_pool)}", flush=True)
            res = self._solve(item["theta"], maxfun=refine_budget, rhobeg=0.05, rhoend=1.5e-5, seek_global_minimum=False)
            refine_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
        refine_results.sort(key=lambda item: item["log10_chi2"])

        polish_results = []
        polish_pool = refine_results[:2]
        print(f"[super-refit] polish pool={len(polish_pool)} budget={polish_budget}", flush=True)
        for i, item in enumerate(polish_pool):
            print(f"[super-refit] polish {i + 1}/{len(polish_pool)}", flush=True)
            res = self._solve(item["theta"], maxfun=polish_budget, rhobeg=0.022, rhoend=8e-7, seek_global_minimum=False)
            polish_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
        polish_results.sort(key=lambda item: item["log10_chi2"])
        best_polished = polish_results[0]

        verify_results = []
        rng = np.random.default_rng(20260324 + 99)
        print(f"[super-refit] stability checks=6 budget={verify_budget}", flush=True)
        for i in range(6):
            jitter = np.clip(best_polished["theta"] + rng.normal(0.0, 0.01 + 0.005 * i, size=self.theta0.shape), 0.0, 1.0)
            print(f"[super-refit] stability {i + 1}/6", flush=True)
            res = self._solve(jitter, maxfun=verify_budget, rhobeg=0.016, rhoend=8e-7, seek_global_minimum=False)
            verify_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
        verify_results.sort(key=lambda item: item["log10_chi2"])
        best_final = verify_results[0] if verify_results[0]["log10_chi2"] < best_polished["log10_chi2"] else best_polished

        elapsed = time.perf_counter() - t0
        params_free = self.denormalize_params(np.asarray(best_final["theta"], dtype=float))
        stability_values = [item["log10_chi2"] for item in verify_results] + [best_polished["log10_chi2"]]

        return {
            "elapsed_s": elapsed,
            "nf": int(
                sum(item["nf"] for item in stage1_results)
                + sum(item["nf"] for item in stage2_results)
                + sum(item["nf"] for item in refine_results)
                + sum(item["nf"] for item in polish_results)
                + sum(item["nf"] for item in verify_results)
            ),
            "flag": int(best_final["flag"]),
            "free_params": params_free.tolist(),
            "log10_chi2": float(best_final["log10_chi2"]),
            "stage1_best_log10_chi2": float(stage1_results[0]["log10_chi2"]),
            "stage2_best_log10_chi2": float(stage2_results[0]["log10_chi2"]),
            "refine_best_log10_chi2": float(refine_results[0]["log10_chi2"]),
            "polish_best_log10_chi2": float(best_polished["log10_chi2"]),
            "stability_min_log10_chi2": float(min(stability_values)),
            "stability_max_log10_chi2": float(max(stability_values)),
            "stability_span_log10_chi2": float(max(stability_values) - min(stability_values)),
            "stability_runs": [
                {"log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in verify_results
            ],
        }


def run_refit(stage2_maxfun: int) -> Path:
    session = SuperBudgetFitSession(CURRENT_SPEC)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(stage2_maxfun=stage2_maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "candidate": CURRENT_SPEC.name,
        "fit_name": CURRENT_SPEC.fit_name,
        "param_names": CURRENT_SPEC.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }

    out_path = RESULTS_DIR / "super_refit_broad_bump_42.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metrics = best_eval["metrics"]
    summary = pd.DataFrame(
        [
            {
                "candidate": CURRENT_SPEC.name,
                "fit_name": CURRENT_SPEC.fit_name,
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2dN_collider": metrics["chi2dN_collider"],
                "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "fit_evals": fit_info["nf"],
                "fit_elapsed_s": fit_info["elapsed_s"],
                "stability_span_log10_chi2": fit_info["stability_span_log10_chi2"],
                **{name: value for name, value in zip(CURRENT_SPEC.param_names, best_eval["full_params"])},
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(best_eval["high_energy_rows"]).to_csv(RESULTS_DIR / "high_energy_rows.csv", index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-maxfun", type=int, default=700)
    args = parser.parse_args()

    out_path = run_refit(stage2_maxfun=args.stage2_maxfun)
    print(f"Wrote {out_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
