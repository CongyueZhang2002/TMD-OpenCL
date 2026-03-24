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


RESULTS_DIR = FITS_DIR / "broad_bump_42_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = FITS_DIR / "aggressive_table_refit_results" / "poly_bstar_cslog_loggauss__4-2.json"

DEFAULT_CANDIDATES = [
    "loggauss_w045",
    "loggauss_w060",
    "loggauss_w080",
    "loggauss_w060_alpha1",
    "logsech_w045",
    "loglorentz_w045",
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
    if variant == "loggauss":
        bump_expr = "amp * expf(-0.5f * u * u)"
    elif variant == "logsech":
        bump_expr = "amp * sechf(u)"
    elif variant == "loglorentz":
        bump_expr = "amp / (1.f + u * u)"
    else:
        raise ValueError(f"Unknown bump variant: {variant}")

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
  float lambda1, lambda2, lambda3, lambda4, alpha;
  float logx0, sigx, amp;
  float BNP, c0, c1;
}} Params_Struct;

inline float clampf(float x, float lo, float hi) {{ return fmin(fmax(x, lo), hi); }}
inline float sechf(float t) {{ t = fabs(t); float v = exp(-2.f * t); return (2.f * exp(-t)) / (1.f + v); }}

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{{
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float alpha = p->alpha;
  const float logx0 = p->logx0;
  const float sigx = p->sigx;
  const float amp = p->amp;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float base = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x);

  float u = (log(x) - logx0) / fmax(sigx, 1e-4f);
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
    base = _best_free_params(BASE_RESULT)
    params = ["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"]

    def bounds(sigx_lo: float) -> list[tuple[float, float]]:
        return [
            (0.02, 8.0),
            (0.02, 8.0),
            (-10.0, 10.0),
            (-0.5, 0.5),
            (0.0, 2.0),
            (np.log(1e-4), np.log(0.3)),
            (sigx_lo, 2.5),
            (-3.0, 3.0),
            (0.4, 4.5),
            (0.0, 0.25),
            (0.0, 0.25),
        ]

    return [
        CandidateSpec(
            name="loggauss_w045",
            fit_name="BroadBump42LogGaussW045",
            np_name="NP-BroadBump42LogGaussW045.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], max(base[6], 0.45), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.45),
            frozen_indices=[],
            kernel_variant="loggauss",
        ),
        CandidateSpec(
            name="loggauss_w060",
            fit_name="BroadBump42LogGaussW060",
            np_name="NP-BroadBump42LogGaussW060.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], max(base[6], 0.60), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.60),
            frozen_indices=[],
            kernel_variant="loggauss",
        ),
        CandidateSpec(
            name="loggauss_w080",
            fit_name="BroadBump42LogGaussW080",
            np_name="NP-BroadBump42LogGaussW080.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], max(base[6], 0.80), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.80),
            frozen_indices=[],
            kernel_variant="loggauss",
        ),
        CandidateSpec(
            name="loggauss_w060_alpha1",
            fit_name="BroadBump42LogGaussW060Alpha1",
            np_name="NP-BroadBump42LogGaussW060Alpha1.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], 1.0, base[5], max(base[6], 0.60), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.60),
            frozen_indices=[4],
            kernel_variant="loggauss",
        ),
        CandidateSpec(
            name="logsech_w045",
            fit_name="BroadBump42LogSechW045",
            np_name="NP-BroadBump42LogSechW045.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], max(base[6], 0.45), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.45),
            frozen_indices=[],
            kernel_variant="logsech",
        ),
        CandidateSpec(
            name="loglorentz_w045",
            fit_name="BroadBump42LogLorentzW045",
            np_name="NP-BroadBump42LogLorentzW045.cl",
            param_names=params,
            initial_params=[base[0], base[1], base[2], base[3], base[4], base[5], max(base[6], 0.45), base[7], base[8], base[9], base[10]],
            bounds=bounds(0.45),
            frozen_indices=[],
            kernel_variant="loglorentz",
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
    ref = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
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
            "delta_total_vs_42_loggauss": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
            "delta_absdev_vs_42_loggauss": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
            "delta_shortfall_vs_42_loggauss": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
            "fit_evals": result["fit"]["nf"],
            "fit_flag": result["fit"]["flag"],
        }
        for pname, value in zip(result["param_names"], result["best"]["full_params"]):
            row[pname] = value
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["chi2dN_total", "highE_mean_absdev_first3", "highE_mean_shortfall_first3"]
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
