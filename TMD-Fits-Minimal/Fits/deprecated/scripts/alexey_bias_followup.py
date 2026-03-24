from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec, FitSession


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "alexey_bias_followup_results"
RESULTS_DIR.mkdir(exist_ok=True)

POWSEED_PATH = FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json"
ART17M2_PATH = FITS_DIR / "art23_family_refine_results" / "art17m2_art23cslog_refine.json"


def render_kernel() -> str:
    return """#define bmax 1.1229189f
#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func(float b, float Q) {
    float mu = bmax / b;
    return max(mu, 1.0f);
}

typedef struct {
  float lambda1, lambda2, l0;
  float BNP, c0, c1;
} Params_Struct;

inline float clampf(float x, float lo, float hi) { return fmin(fmax(x, lo), hi); }

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float l0 = p->l0;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float b2 = b * b;
  float shape = lambda1 * xbar + lambda2 * x;
  float denom = sqrtf(1.f + l0 * x * x * b2);
  float SNP_mu = expf(-shape * b2 / fmax(denom, 1e-12f));

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;

  return (float2)(SNP_mu, SNP_ze);
}

__kernel void NP_f_vec(
    __global const float* x,
    __global const float* b,
    int N,
    __constant Params_Struct* P,
    __global float* out_mu,
    __global float* out_ze
){
    int i = get_global_id(0);
    if (i >= N) return;

    float2 r = NP_f_func(x[i], b[i], P);
    out_mu[i] = r.x;
    out_ze[i] = r.y;
}
"""


def render_card(spec: CandidateSpec) -> str:
    init_vals = ", ".join(f"{x:.12g}" for x in spec.initial_params)
    bounds_vals = ",\n    ".join(f"({lo:.12g}, {hi:.12g})" for lo, hi in spec.bounds)
    struct_fields = "\n".join(f"    {name}::Float32" for name in spec.param_names)
    return f"""#----------------------------------------------------------------------------
# NP
#----------------------------------------------------------------------------

const flavor_scheme = "FI"
const NP_name = "{spec.np_name}"

struct Params_Struct
{struct_fields}
end

initial_params = [{init_vals}]

bounds_raw = [
    {bounds_vals}
]

frozen_indices = []

#----------------------------------------------------------------------------
# PDF
#----------------------------------------------------------------------------

const table_name = "MSHT20N3LO-MC-0-2"
const pdf_name = "approximate"
const error_sets_name = "MSHT20N3LO-MC"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"
"""


def make_spec() -> CandidateSpec:
    data = json.loads(POWSEED_PATH.read_text(encoding="utf-8"))
    p = data["best"]["full_params"]
    spec = CandidateSpec(
        name="alexey_bias_art23cslog",
        fit_name="AlexeyBiasART23CSLog",
        np_name="NP-AlexeyBiasART23CSLog.cl",
        param_names=["lambda1", "lambda2", "l0", "BNP", "c0", "c1"],
        initial_params=[max(p[0], 0.05), max(p[1], 0.05), 1.0, p[6], p[7], p[8]],
        bounds=[(0.02, 4.0), (0.02, 4.0), (0.0, 20.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
        frozen_indices=[],
        kernel_variant="alexey_bias_art23cslog",
    )
    (CARDS_DIR / f"{spec.fit_name}.jl").write_text(render_card(spec), encoding="utf-8")
    (NP_DIR / spec.np_name).write_text(render_kernel(), encoding="utf-8")
    return spec


class AlexeyBiasFitSession(FitSession):
    def candidate_start_points(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = [self.theta0.copy()]
        seed = 20260323 + sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for scale in [0.03, 0.06, 0.10, 0.16]:
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))

        anchor_values = {
            "lambda1": [0.06, 0.18, 0.35, 0.60, 0.85],
            "lambda2": [0.06, 0.18, 0.35, 0.60, 0.85],
            "l0": [0.0, 0.08, 0.18, 0.35, 0.60, 0.85],
            "BNP": [0.10, 0.30, 0.55, 0.80],
            "c0": [0.05, 0.20, 0.40, 0.70],
            "c1": [0.05, 0.20, 0.40, 0.70],
        }
        for pname, vals in anchor_values.items():
            idx = self.free_param_names.index(pname)
            for val in vals:
                start = self.theta0.copy()
                start[idx] = val
                starts.append(start)

        for _ in range(6):
            starts.append(np.clip(rng.uniform(0.0, 1.0, size=self.theta0.shape), 0.0, 1.0))

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique[:18]

    def fit(self, maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1_budget = max(70, maxfun // 4)
        stage2_budget = maxfun
        refine_budget = max(140, maxfun // 2)
        polish_budget = max(140, maxfun // 3)

        stage1_results = []
        for i, start in enumerate(self.candidate_start_points()):
            res = self._solve(start, maxfun=stage1_budget, rhobeg=0.18, rhoend=8e-4, seek_global_minimum=True)
            stage1_results.append({"start_id": i, "theta": np.asarray(res.x, dtype=float), "log10_chi2": float(res.f), "nf": int(res.nf), "flag": int(res.flag)})
        stage1_results.sort(key=lambda item: item["log10_chi2"])

        stage2_results = []
        for item in stage1_results[:4]:
            res = self._solve(item["theta"], maxfun=stage2_budget, rhobeg=0.10, rhoend=1e-4, seek_global_minimum=True)
            stage2_results.append({"theta": np.asarray(res.x, dtype=float), "log10_chi2": float(res.f), "nf": int(res.nf), "flag": int(res.flag)})
        stage2_results.sort(key=lambda item: item["log10_chi2"])

        refine_results = []
        for item in stage2_results[:2]:
            res = self._solve(item["theta"], maxfun=refine_budget, rhobeg=0.05, rhoend=2e-5, seek_global_minimum=False)
            refine_results.append({"theta": np.asarray(res.x, dtype=float), "log10_chi2": float(res.f), "nf": int(res.nf), "flag": int(res.flag)})
        refine_results.sort(key=lambda item: item["log10_chi2"])

        best_refined = refine_results[0]
        res = self._solve(best_refined["theta"], maxfun=polish_budget, rhobeg=0.025, rhoend=1e-6, seek_global_minimum=False)

        elapsed = time.perf_counter() - t0
        params_free = self.denormalize_params(np.asarray(res.x, dtype=float))
        return {
            "elapsed_s": elapsed,
            "nf": int(sum(item["nf"] for item in stage1_results) + sum(item["nf"] for item in stage2_results) + sum(item["nf"] for item in refine_results) + int(res.nf)),
            "flag": int(res.flag),
            "free_params": params_free.tolist(),
            "log10_chi2": float(res.f),
            "stage1": [{"start_id": item["start_id"], "log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]} for item in stage1_results],
            "stage2": [{"log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]} for item in stage2_results],
            "refine": [{"log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]} for item in refine_results],
            "polish_nf": int(res.nf),
        }


def run_candidate(maxfun: int) -> Path:
    spec = make_spec()
    session = AlexeyBiasFitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
    result = {
        "candidate": spec.name,
        "fit_name": spec.fit_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }
    out_path = RESULTS_DIR / f"{spec.name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> pd.DataFrame:
    rows = []
    ref = json.loads(POWSEED_PATH.read_text(encoding="utf-8"))["best"]["metrics"]
    rows.append({
        "candidate": "powerseed",
        **ref,
    })
    if ART17M2_PATH.exists():
        art = json.loads(ART17M2_PATH.read_text(encoding="utf-8"))["best"]["metrics"]
        rows.append({
            "candidate": "art17m2_art23cslog_refine",
            **art,
        })
    path = RESULTS_DIR / "alexey_bias_art23cslog.json"
    if path.exists():
        alex = json.loads(path.read_text(encoding="utf-8"))["best"]["metrics"]
        rows.append({
            "candidate": "alexey_bias_art23cslog",
            **alex,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        keep = [
            "candidate",
            "chi2dN_total",
            "chi2dN_collider",
            "chi2dN_fixed_target",
            "highE_mean_absdev_first3",
            "highE_mean_shortfall_first3",
            "cms_highmass_mean_signed_first3",
            "zlike_mean_signed_first3",
            "compute_s",
        ]
        df = df[keep]
        df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def orchestrate(maxfun: int) -> None:
    cmd = [str(PYTHON), str(Path(__file__)), "--worker", "--maxfun", str(maxfun)]
    proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
    print(f"worker exit code {proc.returncode}")
    df = summarize()
    if not df.empty:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--maxfun", type=int, default=320)
    args = parser.parse_args()

    if args.worker:
        out_path = run_candidate(maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    orchestrate(maxfun=args.maxfun)


if __name__ == "__main__":
    main()
