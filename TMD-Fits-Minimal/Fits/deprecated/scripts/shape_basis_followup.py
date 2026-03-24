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


RESULTS_DIR = FITS_DIR / "shape_basis_followup_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json"

DEFAULT_CANDIDATES = [
    "basis_powerseed_baseline",
    "basis_sqrtx_log",
    "basis_x_softlog",
    "basis_sqrtx_softlog",
]


def _base_best() -> list[float]:
    data = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    return list(map(float, data["best"]["free_params"]))


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


def _render_kernel(spec: CandidateSpec) -> str:
    variant = spec.kernel_variant
    if variant == "basis_powerseed_baseline":
        struct_fields = "  float lambda1, lambda2, lambda4;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda4 = p->lambda4;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float base = lambda1 * xbar + lambda2 * x + lambda4 * log(x);
"""
    elif variant == "basis_sqrtx_log":
        struct_fields = "  float lambda1, lambda2, lambda4;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda4 = p->lambda4;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float z = sqrtf(x);
  float zbar = 1.f - z;
  float base = lambda1 * zbar + lambda2 * z + lambda4 * log(x);
"""
    elif variant == "basis_x_softlog":
        struct_fields = "  float lambda1, lambda2, lambda4;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda4 = p->lambda4;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float x0 = expf(logx0);
  float softlog = log(x + x0) - log(1.f + x0);
  float base = lambda1 * xbar + lambda2 * x + lambda4 * softlog;
"""
    elif variant == "basis_sqrtx_softlog":
        struct_fields = "  float lambda1, lambda2, lambda4;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda4 = p->lambda4;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float z = sqrtf(x);
  float zbar = 1.f - z;
  float x0 = expf(logx0);
  float softlog = log(x + x0) - log(1.f + x0);
  float base = lambda1 * zbar + lambda2 * z + lambda4 * softlog;
"""
    else:
        raise ValueError(variant)

    return f"""#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f

#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func(float b, float Q) {{
    float mu = bmax / b;
    return max(mu, 1.0f);
}}

typedef struct {{
{struct_fields}}} Params_Struct;

inline float clampf(float x, float lo, float hi) {{ return fmin(fmax(x, lo), hi); }}
inline float sechf(float t) {{ t = fabs(t); float u = exp(-2.f * t); return (2.f * exp(-t)) / (1.f + u); }}

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{{
{body}
  float lx = log(x) - logx0;
  float invsig = 1.f / fmax(sigx, 1e-4f);
  float u = lx * invsig;
  float bump = amp * expf(-0.5f * u * u);
  float shape = base + bump;
  float SNP_mu = sechf(shape * b);

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;

  return (float2)(SNP_mu, SNP_ze);
}}

__kernel void NP_f_vec(
    __global const float* x,
    __global const float* b,
    int N,
    __constant Params_Struct* P,
    __global float* out_mu,
    __global float* out_ze
){{
    int i = get_global_id(0);
    if (i >= N) return;

    float2 r = NP_f_func(x[i], b[i], P);
    out_mu[i] = r.x;
    out_ze[i] = r.y;
}}
"""


def followup_specs() -> list[CandidateSpec]:
    lam1, lam2, lam4, logx0, sigx, amp, BNP, c0, c1 = _base_best()
    common_bounds = [
        (0.02, 8.0),
        (0.02, 8.0),
        (-0.5, 0.5),
        (np.log(1e-4), np.log(0.3)),
        (0.15, 2.5),
        (-2.0, 2.0),
        (0.4, 4.5),
        (0.0, 0.25),
        (0.0, 0.25),
    ]
    common_init = [lam1, lam2, lam4, logx0, sigx, amp, BNP, c0, c1]
    common_params = ["lambda1", "lambda2", "lambda4", "logx0", "sigx", "amp", "BNP", "c0", "c1"]

    return [
        CandidateSpec(
            name="basis_powerseed_baseline",
            fit_name="BasisPowerSeedBaseline",
            np_name="NP-BasisPowerSeedBaseline.cl",
            param_names=common_params,
            initial_params=common_init,
            bounds=common_bounds,
            frozen_indices=[],
            kernel_variant="basis_powerseed_baseline",
        ),
        CandidateSpec(
            name="basis_sqrtx_log",
            fit_name="BasisSqrtXLog",
            np_name="NP-BasisSqrtXLog.cl",
            param_names=common_params,
            initial_params=common_init,
            bounds=common_bounds,
            frozen_indices=[],
            kernel_variant="basis_sqrtx_log",
        ),
        CandidateSpec(
            name="basis_x_softlog",
            fit_name="BasisXSoftLog",
            np_name="NP-BasisXSoftLog.cl",
            param_names=common_params,
            initial_params=common_init,
            bounds=common_bounds,
            frozen_indices=[],
            kernel_variant="basis_x_softlog",
        ),
        CandidateSpec(
            name="basis_sqrtx_softlog",
            fit_name="BasisSqrtXSoftLog",
            np_name="NP-BasisSqrtXSoftLog.cl",
            param_names=common_params,
            initial_params=common_init,
            bounds=common_bounds,
            frozen_indices=[],
            kernel_variant="basis_sqrtx_softlog",
        ),
    ]


def write_followup_candidate_files() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in followup_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
        (NP_DIR / spec.np_name).write_text(_render_kernel(spec), encoding="utf-8")
    return specs


def run_candidate(spec: CandidateSpec, maxfun: int) -> Path:
    session = FitSession(spec)
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


def summarize_results(specs: dict[str, CandidateSpec]) -> pd.DataFrame:
    ref = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]
    rows = []
    for name in specs:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["best"]["metrics"]
        row = {
            "candidate": name,
            "chi2dN_total": metrics["chi2dN_total"],
            "chi2dN_collider": metrics["chi2dN_collider"],
            "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
            "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
            "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
            "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
            "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
            "delta_total_vs_powerseed": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
            "delta_absdev_vs_powerseed": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
            "delta_shortfall_vs_powerseed": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
            "fit_evals": result["fit"]["nf"],
            "fit_flag": result["fit"]["flag"],
        }
        best_full = result["best"]["full_params"]
        for pname, value in zip(result["param_names"], best_full):
            row[pname] = value
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["highE_mean_absdev_first3", "highE_mean_shortfall_first3", "chi2dN_total"]
    ).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def run_worker(candidate: str, maxfun: int) -> None:
    specs = write_followup_candidate_files()
    run_candidate(specs[candidate], maxfun=maxfun)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(candidates: list[str], maxfun: int) -> None:
    specs = write_followup_candidate_files()
    for name in candidates:
        cmd = [str(PYTHON), str(Path(__file__)), "--candidate", name, "--maxfun", str(maxfun)]
        print(f"\n=== Running {name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{name}: exit code {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {name}")

    print("\n=== Summary ===")
    df = summarize_results(specs)
    if not df.empty:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str)
    parser.add_argument("--maxfun", type=int, default=150)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    if args.candidate:
        run_worker(args.candidate, maxfun=args.maxfun)
        return

    orchestrate(args.candidates, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
