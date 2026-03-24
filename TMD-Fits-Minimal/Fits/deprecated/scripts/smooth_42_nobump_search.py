from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from art23_family_search import Art23FamilyFitSession
from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec


RESULTS_DIR = FITS_DIR / "smooth_42_nobump_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_42_RESULT = FITS_DIR / "aggressive_table_refit_results" / "poly_bstar_cslog__4-2.json"
DEFAULT_CANDIDATES = [
    "poly_bstar_cslog_42_base",
    "poly_bstar_cslog_42_alpha1",
    "poly_bstar_cslog_42_nolog",
    "poly_bstar_cslog2_42",
    "art23_mu_poly_cslog_42",
    "art17m1_art23cslog_42",
    "art17m2_art23cslog_42",
    "sqrtx_poly_bstar_cslog_42",
]


def _best_free_params(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
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

const table_name = "MSHT20N3LO-MC-4-2"
const pdf_name = "approximate"
const error_sets_name = "MSHT20N3LO-MC"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"
"""


def _render_kernel(spec: CandidateSpec) -> str:
    variant = spec.kernel_variant

    if variant == "poly_bstar_cslog":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4, alpha;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float shape = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x);

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
"""
    elif variant == "poly_bstar_cslog2":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4, alpha;\n  float BNP, c0, c1, c2;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;
  const float c2 = p->c2;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float shape = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x);

  float t = b / bmax;
  float t2 = t * t;
  float t4 = t2 * t2;
  float bstar_mu = b * powr(1.f + t4, 0.25f * (alpha - 1.f));
  float SNP_mu = sechf(shape * bstar_mu);

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio + c2 * log_ratio * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "art23_mu_poly_cslog":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float shape = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x);
  float SNP_mu = sechf(shape * b);

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "art17m1_art23cslog":
        struct_fields = "  float lambda1, lambda2, BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  float inv_l1 = 1.f / fmax(lambda1, 1e-6f);
  float A = lambda2 * inv_l1 - 0.5f * lambda1;
  float B = lambda2 * inv_l1 + 0.5f * lambda1;
  float SNP_mu = expf(logcoshf(A * b) - logcoshf(B * b));

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "art17m2_art23cslog":
        struct_fields = "  float lambda1, lambda2, BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f);
  float b2 = b * b;
  float ratio = (lambda2 * lambda2) / fmax(lambda1 * lambda1, 1e-12f);
  float denom = sqrtf(1.f + x * x * b2 * ratio);
  float SNP_mu = expf(-lambda2 * x * b2 / fmax(denom, 1e-12f));

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "sqrtx_poly_bstar_cslog":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4, alpha;\n  float BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float z = sqrtf(x);
  float zbar = 1.f - z;
  float zzbar = z * zbar;
  float shape = lambda1 * zbar + lambda2 * z + lambda3 * zzbar + lambda4 * log(x);

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
"""
    else:
        raise ValueError(f"Unknown kernel variant: {variant}")

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
inline float logcoshf(float t) {{ t = fabs(t); return t + log(1.f + exp(-2.f * t)) - 0.6931471805599453f; }}

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{{
{body}
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
    base = _best_free_params(BASE_42_RESULT)
    art17m1 = [0.09535798263172778, 0.09745060199812608, 1.45, 0.07419699259000759, 0.021507110056008586]
    art17m2 = [0.07813285426341661, 1.5017464623554853, 1.45, 0.07419699259000759, 0.021507110056008586]
    mu_poly = [base[0], base[1], base[2], base[3], base[5], base[6], base[7]]

    bounds_poly = [
        (0.02, 8.0),
        (0.02, 8.0),
        (-10.0, 10.0),
        (-0.5, 0.5),
        (0.0, 2.0),
        (0.4, 4.5),
        (0.0, 0.25),
        (0.0, 0.25),
    ]
    params_poly = ["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "BNP", "c0", "c1"]

    bounds_poly2 = bounds_poly + [(0.0, 0.25)]
    params_poly2 = params_poly + ["c2"]

    bounds_mu_poly = [
        (0.02, 8.0),
        (0.02, 8.0),
        (-10.0, 10.0),
        (-0.5, 0.5),
        (0.4, 4.5),
        (0.0, 0.25),
        (0.0, 0.25),
    ]
    params_mu_poly = ["lambda1", "lambda2", "lambda3", "lambda4", "BNP", "c0", "c1"]

    bounds_art17 = [(0.02, 1.5), (0.01, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)]
    params_art17 = ["lambda1", "lambda2", "BNP", "c0", "c1"]

    return [
        CandidateSpec(
            name="poly_bstar_cslog_42_base",
            fit_name="Smooth42NoBumpPolyBstarCSLog",
            np_name="NP-Smooth42NoBumpPolyBstarCSLog.cl",
            param_names=params_poly,
            initial_params=base,
            bounds=bounds_poly,
            frozen_indices=[],
            kernel_variant="poly_bstar_cslog",
        ),
        CandidateSpec(
            name="poly_bstar_cslog_42_alpha1",
            fit_name="Smooth42NoBumpPolyBstarCSLogAlpha1",
            np_name="NP-Smooth42NoBumpPolyBstarCSLogAlpha1.cl",
            param_names=params_poly,
            initial_params=[base[0], base[1], base[2], base[3], 1.0, base[5], base[6], base[7]],
            bounds=bounds_poly,
            frozen_indices=[4],
            kernel_variant="poly_bstar_cslog",
        ),
        CandidateSpec(
            name="poly_bstar_cslog_42_nolog",
            fit_name="Smooth42NoBumpPolyBstarCSLogNoLog",
            np_name="NP-Smooth42NoBumpPolyBstarCSLogNoLog.cl",
            param_names=params_poly,
            initial_params=[base[0], base[1], base[2], 0.0, base[4], base[5], base[6], base[7]],
            bounds=bounds_poly,
            frozen_indices=[3],
            kernel_variant="poly_bstar_cslog",
        ),
        CandidateSpec(
            name="poly_bstar_cslog2_42",
            fit_name="Smooth42NoBumpPolyBstarCSLog2",
            np_name="NP-Smooth42NoBumpPolyBstarCSLog2.cl",
            param_names=params_poly2,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], base[6], base[7], 0.0],
            bounds=bounds_poly2,
            frozen_indices=[],
            kernel_variant="poly_bstar_cslog2",
        ),
        CandidateSpec(
            name="art23_mu_poly_cslog_42",
            fit_name="Smooth42NoBumpMuPolyCSLog",
            np_name="NP-Smooth42NoBumpMuPolyCSLog.cl",
            param_names=params_mu_poly,
            initial_params=mu_poly,
            bounds=bounds_mu_poly,
            frozen_indices=[],
            kernel_variant="art23_mu_poly_cslog",
        ),
        CandidateSpec(
            name="art17m1_art23cslog_42",
            fit_name="Smooth42NoBumpART17M1CSLog",
            np_name="NP-Smooth42NoBumpART17M1CSLog.cl",
            param_names=params_art17,
            initial_params=art17m1,
            bounds=bounds_art17,
            frozen_indices=[],
            kernel_variant="art17m1_art23cslog",
        ),
        CandidateSpec(
            name="art17m2_art23cslog_42",
            fit_name="Smooth42NoBumpART17M2CSLog",
            np_name="NP-Smooth42NoBumpART17M2CSLog.cl",
            param_names=params_art17,
            initial_params=art17m2,
            bounds=bounds_art17,
            frozen_indices=[],
            kernel_variant="art17m2_art23cslog",
        ),
        CandidateSpec(
            name="sqrtx_poly_bstar_cslog_42",
            fit_name="Smooth42NoBumpSqrtXPolyBstarCSLog",
            np_name="NP-Smooth42NoBumpSqrtXPolyBstarCSLog.cl",
            param_names=params_poly,
            initial_params=base,
            bounds=bounds_poly,
            frozen_indices=[],
            kernel_variant="sqrtx_poly_bstar_cslog",
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
    rows = []
    ref = json.loads(BASE_42_RESULT.read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]

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
            "delta_total_vs_poly_bstar_42": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
            "delta_absdev_vs_poly_bstar_42": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
            "delta_shortfall_vs_poly_bstar_42": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
            "fit_evals": result["fit"]["nf"],
            "fit_flag": result["fit"]["flag"],
        }
        for pname, value in zip(result["param_names"], result["best"]["full_params"]):
            row[pname] = value
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["highE_mean_absdev_first3", "highE_mean_shortfall_first3", "chi2dN_total"]
    ).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def run_worker(candidate: str, maxfun: int) -> None:
    specs = write_candidate_files()
    run_candidate(specs[candidate], maxfun=maxfun)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(candidates: list[str], maxfun: int) -> None:
    specs = write_candidate_files()
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
    parser.add_argument("--maxfun", type=int, default=320)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    if args.candidate:
        run_worker(args.candidate, maxfun=args.maxfun)
        return

    orchestrate(args.candidates, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
