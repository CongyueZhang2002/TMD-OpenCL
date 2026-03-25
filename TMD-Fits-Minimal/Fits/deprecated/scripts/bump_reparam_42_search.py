from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from art23_family_search import Art23FamilyFitSession
from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, CandidateSpec


RESULTS_DIR = FITS_DIR / "bump_reparam_42_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = FITS_DIR / "broad_bump_42_prune_results" / "broad_bump_42_alpha1_lambda2_zero.json"
DEFAULT_CANDIDATES = [
    "area_gauss",
    "area_lorentz",
    "fixed_gauss_basis",
]


def _best_full_params(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(map(float, data["best"]["full_params"]))


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

const table_name = "MSHT20N3LO-MC-4-2"
const pdf_name = "approximate"
const error_sets_name = "MSHT20N3LO-MC"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"
"""


def _render_kernel(spec: CandidateSpec) -> str:
    if spec.kernel_variant == "area_gauss":
        struct_fields = "lambda1, lambda3, lambda4, alpha, logx0, sigx, amp, BNP, c0, c1"
        loads = """  const float lambda1 = p->lambda1;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;"""
        bump_expr = "amp * expf(-0.5f * u * u) / fmax(sigx, 1e-4f)"
    elif spec.kernel_variant == "area_lorentz":
        struct_fields = "lambda1, lambda3, lambda4, alpha, logx0, sigx, amp, BNP, c0, c1"
        loads = """  const float lambda1 = p->lambda1;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;"""
        bump_expr = "amp / (fmax(sigx, 1e-4f) * (1.f + u * u))"
    elif spec.kernel_variant == "fixed_gauss_basis":
        struct_fields = "lambda1, lambda3, lambda4, alpha, amp, BNP, c0, c1"
        loads = """  const float lambda1 = p->lambda1;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;"""
        bump_expr = "amp * expf(-0.5f * u * u)"
    else:
        raise ValueError(f"Unknown kernel variant: {spec.kernel_variant}")

    fixed_bump_consts = ""
    u_expr = "float u = (logx - logx0) / fmax(sigx, 1e-4f);"
    if spec.kernel_variant == "fixed_gauss_basis":
        fixed_bump_consts = "#define FIXED_LOGX0 (-4.605170186f)\n#define FIXED_SIGX (0.8f)\n"
        u_expr = "float u = (logx - FIXED_LOGX0) / FIXED_SIGX;"

    return f"""#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f
{fixed_bump_consts}
#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func(float b, float Q) {{
    float mu = bmax / b;
    return max(mu, 1.0f);
}}

typedef struct {{
  float {struct_fields};
}} Params_Struct;

inline float clampf(float x, float lo, float hi) {{ return fmin(fmax(x, lo), hi); }}
inline float sechf(float t) {{ t = fabs(t); float v = exp(-2.f * t); return (2.f * exp(-t)) / (1.f + v); }}

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{{
{loads}

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float logx = log(x);
  float base = lambda1 * xbar + lambda3 * xxbar + lambda4 * logx;
  {u_expr}
  float bump = {bump_expr};
  float shape = base + bump;

  float t = b / bmax;
  float t2 = t * t;
  float t4 = t2 * t2;
  float bstar_mu = b * powr(1.f + t4, 0.25f * (alpha - 1.f));
  float SNP_mu = sechf(shape * bstar_mu);

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


def candidate_specs() -> list[CandidateSpec]:
    base = _best_full_params(BASE_RESULT)
    # lambda1, lambda2, lambda3, lambda4, alpha, logx0, sigx, amp, BNP, c0, c1
    lambda1, _, lambda3, lambda4, alpha, logx0, sigx, amp, bnp, c0, c1 = base
    area_amp = amp * sigx
    return [
        CandidateSpec(
            name="area_gauss",
            fit_name="BroadBump42AreaGauss",
            np_name="NP-BroadBump42AreaGauss.cl",
            param_names=["lambda1", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"],
            initial_params=[lambda1, lambda3, lambda4, alpha, logx0, sigx, area_amp, bnp, c0, c1],
            bounds=[(0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (np.log(1e-4), np.log(0.3)), (0.6, 2.5), (-6.0, 6.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[3],
            kernel_variant="area_gauss",
        ),
        CandidateSpec(
            name="area_lorentz",
            fit_name="BroadBump42AreaLorentz",
            np_name="NP-BroadBump42AreaLorentz.cl",
            param_names=["lambda1", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"],
            initial_params=[lambda1, lambda3, lambda4, alpha, logx0, sigx, area_amp, bnp, c0, c1],
            bounds=[(0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (np.log(1e-4), np.log(0.3)), (0.6, 2.5), (-6.0, 6.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[3],
            kernel_variant="area_lorentz",
        ),
        CandidateSpec(
            name="fixed_gauss_basis",
            fit_name="BroadBump42FixedGaussBasis",
            np_name="NP-BroadBump42FixedGaussBasis.cl",
            param_names=["lambda1", "lambda3", "lambda4", "alpha", "amp", "BNP", "c0", "c1"],
            initial_params=[lambda1, lambda3, lambda4, alpha, amp, bnp, c0, c1],
            bounds=[(0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (-3.0, 3.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[3],
            kernel_variant="fixed_gauss_basis",
        ),
    ]


def write_candidate_files() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in candidate_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
        (NP_DIR / spec.np_name).write_text(_render_kernel(spec), encoding="utf-8")
    return specs


def run_candidate(spec: CandidateSpec, maxfun: int) -> Path:
    session = Art23FamilyFitSession(spec)
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
    rows = [
        {
            "candidate": "current_best",
            "fit_name": "BroadBump42LogGaussAlpha1NoLambda2",
            "chi2dN_total": ref_metrics["chi2dN_total"],
            "highE_mean_absdev_first3": ref_metrics["highE_mean_absdev_first3"],
            "highE_mean_shortfall_first3": ref_metrics["highE_mean_shortfall_first3"],
        }
    ]
    for name, spec in specs.items():
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["best"]["metrics"]
        rows.append(
            {
                "candidate": name,
                "fit_name": spec.fit_name,
                "chi2dN_total": metrics["chi2dN_total"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "delta_chi2dN_vs_current": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
                "delta_absdev_vs_current": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
                "delta_shortfall_vs_current": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
                "fit_evals": data["fit"]["nf"],
                "fit_elapsed_s": data["fit"]["elapsed_s"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--maxfun", type=int, default=220)
    args = parser.parse_args()

    specs = write_candidate_files()
    for name in args.candidates:
        run_candidate(specs[name], maxfun=args.maxfun)
    summarize_results({name: specs[name] for name in args.candidates})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
