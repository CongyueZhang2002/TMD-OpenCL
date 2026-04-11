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


RESULTS_DIR = RESULTS_ROOT / "deep_refit_current_best_42_results"
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


CURRENT_SPEC = CandidateSpec(
    name="broad_bump_42_alpha1_nolambda2_currentbest_deep",
    fit_name="BroadBump42LogGaussAlpha1NoLambda2",
    np_name="NP-BroadBump42LogGaussAlpha1NoLambda2.cl",
    param_names=PARAM_NAMES,
    initial_params=[
        0.04375743605,
        0.9774184714,
        -1.960659735,
        -4.99216975,
        0.717326296,
        -0.338498232,
        1.455523349,
        0.0717885484,
        0.02583083271,
    ],
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


class DeepBudgetFitSession(Art23FamilyFitSession):
    def candidate_start_points(self) -> list[np.ndarray]:
        starts = [self.theta0.copy()]
        seed = 20260324 + sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for scale in [0.02, 0.04, 0.07, 0.10, 0.14, 0.18, 0.24]:
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))

        anchor_values = {
            "lambda1": [0.10, 0.30, 0.50, 0.70, 0.90],
            "lambda2": [0.05, 0.20, 0.40, 0.65, 0.90],
            "lambda3": [0.10, 0.30, 0.50, 0.70, 0.90],
            "logx0": [0.20, 0.35, 0.50, 0.65, 0.80],
            "sigx": [0.10, 0.25, 0.45, 0.65, 0.85],
            "amp": [0.20, 0.40, 0.60, 0.80],
            "BNP": [0.10, 0.30, 0.50, 0.70, 0.90],
            "c0": [0.05, 0.20, 0.40, 0.65, 0.85],
            "c1": [0.05, 0.20, 0.40, 0.65, 0.85],
        }
        for pname, vals in anchor_values.items():
            if pname not in self.free_param_names:
                continue
            idx = self.free_param_names.index(pname)
            for val in vals:
                start = self.theta0.copy()
                start[idx] = val
                starts.append(start)

        for _ in range(12):
            starts.append(np.clip(rng.uniform(0.0, 1.0, size=self.theta0.shape), 0.0, 1.0))

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique[:28]

    def fit(self, maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1_budget = max(90, maxfun // 4)
        stage2_budget = maxfun
        refine_budget = max(220, maxfun // 2)
        polish_budget = max(180, maxfun // 3)

        stage1_results = []
        starts = self.candidate_start_points()
        print(f"[deep-refit] stage1 starts={len(starts)} budget={stage1_budget}", flush=True)
        for i, start in enumerate(starts):
            print(f"[deep-refit] stage1 {i + 1}/{len(starts)}", flush=True)
            res = self._solve(start, maxfun=stage1_budget, rhobeg=0.16, rhoend=8e-4, seek_global_minimum=True)
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

        stage2_results = []
        stage2_pool = stage1_results[:5]
        print(f"[deep-refit] stage2 pool={len(stage2_pool)} budget={stage2_budget}", flush=True)
        for i, item in enumerate(stage2_pool):
            print(f"[deep-refit] stage2 {i + 1}/{len(stage2_pool)}", flush=True)
            res = self._solve(item["theta"], maxfun=stage2_budget, rhobeg=0.09, rhoend=1e-4, seek_global_minimum=True)
            stage2_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )

        stage2_results.sort(key=lambda item: item["log10_chi2"])

        refine_results = []
        refine_pool = stage2_results[:3]
        print(f"[deep-refit] refine pool={len(refine_pool)} budget={refine_budget}", flush=True)
        for i, item in enumerate(refine_pool):
            print(f"[deep-refit] refine {i + 1}/{len(refine_pool)}", flush=True)
            res = self._solve(item["theta"], maxfun=refine_budget, rhobeg=0.045, rhoend=2e-5, seek_global_minimum=False)
            refine_results.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )

        refine_results.sort(key=lambda item: item["log10_chi2"])
        best_refined = refine_results[0]

        print(f"[deep-refit] polish budget={polish_budget}", flush=True)
        res = self._solve(best_refined["theta"], maxfun=polish_budget, rhobeg=0.02, rhoend=1e-6, seek_global_minimum=False)

        elapsed = time.perf_counter() - t0
        params_free = self.denormalize_params(np.asarray(res.x, dtype=float))
        return {
            "elapsed_s": elapsed,
            "nf": int(
                sum(item["nf"] for item in stage1_results)
                + sum(item["nf"] for item in stage2_results)
                + sum(item["nf"] for item in refine_results)
                + int(res.nf)
            ),
            "flag": int(res.flag),
            "free_params": params_free.tolist(),
            "log10_chi2": float(res.f),
            "stage1": [
                {"start_id": item["start_id"], "log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in stage1_results
            ],
            "stage2": [
                {"log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in stage2_results
            ],
            "refine": [
                {"log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in refine_results
            ],
            "polish_nf": int(res.nf),
        }


def run_refit(maxfun: int) -> Path:
    session = DeepBudgetFitSession(CURRENT_SPEC)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
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

    out_path = RESULTS_DIR / "deep_refit_current_best_42.json"
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
                **{name: value for name, value in zip(CURRENT_SPEC.param_names, best_eval["full_params"])},
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(best_eval["high_energy_rows"]).to_csv(RESULTS_DIR / "high_energy_rows.csv", index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxfun", type=int, default=420)
    args = parser.parse_args()

    out_path = run_refit(maxfun=args.maxfun)
    print(f"Wrote {out_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
