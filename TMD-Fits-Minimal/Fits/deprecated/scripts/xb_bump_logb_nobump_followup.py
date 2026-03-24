from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec, FitSession


RESULTS_DIR = FITS_DIR / "xb_bump_logb_nobump_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = FITS_DIR / "xb_localized_followup_results" / "xb_bump_logb_mult.json"
POWERSEED_RESULT = FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json"


def _base_best() -> dict[str, float]:
    data = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    names = data["param_names"]
    vals = data["best"]["full_params"]
    return {name: float(val) for name, val in zip(names, vals)}


def _render_card(spec: CandidateSpec) -> str:
    struct_fields = "\n".join(f"    {name}::Float32" for name in spec.param_names)
    init_vals = ", ".join(f"{x:.10g}" for x in spec.initial_params)
    bounds_vals = ",\n    ".join(f"({lo:.10g}, {hi:.10g})" for lo, hi in spec.bounds)
    frozen_vals = ", ".join(str(i) for i in spec.frozen_indices)
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

frozen_indices = [{frozen_vals}]

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


def _render_kernel() -> str:
    return """#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f

#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func(float b, float Q) {
    float mu = bmax / b;
    return max(mu, 1.0f);
}

typedef struct {
  float lambda1, lambda2, lambda4;
  float logx0, sigx;
  float logb0, sigb, axb;
  float BNP, c0, c1;
} Params_Struct;

inline float clampf(float x, float lo, float hi) { return fmin(fmax(x, lo), hi); }
inline float sechf(float t) { t = fabs(t); float u = exp(-2.f * t); return (2.f * exp(-t)) / (1.f + u); }

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda4 = p->lambda4;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float logb0 = p->logb0;
  const float sigb = p->sigb;
  const float axb = p->axb;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float base = lambda1 * xbar + lambda2 * x + lambda4 * log(x);

  float lx = log(x) - logx0;
  float ux = lx / fmax(sigx, 1e-4f);
  float wx = expf(-0.5f * ux * ux);

  float blog = log(fmax(b, 1e-4f));
  float lb = blog - logb0;
  float ub = lb / fmax(sigb, 1e-4f);
  float wb = expf(-0.5f * ub * ub);

  float SNP_mu0 = sechf(base * b);
  float expo = clampf(axb * wx * wb, -2.0f, 4.0f);
  float SNP_mu = SNP_mu0 * expf(-expo);

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


def spec() -> CandidateSpec:
    p = _base_best()
    return CandidateSpec(
        name="xb_bump_logb_nobump",
        fit_name="XBBumpLogBNoBump",
        np_name="NP-XBBumpLogBNoBump.cl",
        param_names=["lambda1", "lambda2", "lambda4", "logx0", "sigx", "logb0", "sigb", "axb", "BNP", "c0", "c1"],
        initial_params=[p["lambda1"], p["lambda2"], p["lambda4"], p["logx0"], p["sigx"], p["logb0"], p["sigb"], p["axb"], p["BNP"], p["c0"], p["c1"]],
        bounds=[
            (0.02, 8.0),
            (0.02, 8.0),
            (-0.5, 0.5),
            (np.log(1e-4), np.log(0.3)),
            (0.15, 2.5),
            (np.log(0.08), np.log(4.0)),
            (0.12, 1.8),
            (-2.0, 2.0),
            (0.4, 4.5),
            (0.0, 0.25),
            (0.0, 0.25),
        ],
        frozen_indices=[],
        kernel_variant="xb_bump_logb_nobump",
    )


def write_files() -> CandidateSpec:
    candidate = spec()
    (CARDS_DIR / f"{candidate.fit_name}.jl").write_text(_render_card(candidate), encoding="utf-8")
    (NP_DIR / candidate.np_name).write_text(_render_kernel(), encoding="utf-8")
    return candidate


def run_fit(candidate: CandidateSpec, maxfun: int) -> Path:
    session = FitSession(candidate)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "candidate": candidate.name,
        "fit_name": candidate.fit_name,
        "param_names": candidate.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }
    out_path = RESULTS_DIR / f"{candidate.name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> pd.DataFrame:
    rows = []
    for name, path in {
        "powerseed": POWERSEED_RESULT,
        "xb_bump_logb_mult": BASE_RESULT,
        "xb_bump_logb_nobump": RESULTS_DIR / "xb_bump_logb_nobump.json",
    }.items():
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        metrics = obj["best"]["metrics"]
        rows.append(
            {
                "candidate": name,
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2dN_collider": metrics["chi2dN_collider"],
                "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxfun", type=int, default=180)
    args = parser.parse_args()

    candidate = write_files()
    run_fit(candidate, maxfun=args.maxfun)
    df = summarize()
    if not df.empty:
        print(df.to_string(index=False))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
