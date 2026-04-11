from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec, write_candidate_files as write_auto_candidate_files
from art23_family_search import Art23FamilyFitSession, RESULTS_DIR as FAMILY_RESULTS_DIR, _render_card, _render_kernel, write_family_candidate_files
from art23_family_refine import write_refine_candidate_files
from _paths import RESULTS_ROOT


RESULTS_DIR = RESULTS_ROOT / "localized_shape_followup_results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_CANDIDATES = [
    "poly_bstar_cslog_loggauss",
    "poly_bstar_cslog_loggauss_odd",
]


def _best(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(map(float, data["fit"]["free_params"]))


def _render_kernel_followup(spec: CandidateSpec) -> str:
    if spec.kernel_variant == "poly_bstar_cslog_loggauss":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4, alpha;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
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

  float lx = log(x) - logx0;
  float invsig = 1.f / fmax(sigx, 1e-4f);
  float u = lx * invsig;
  float bump = amp * expf(-0.5f * u * u);
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
"""
    elif spec.kernel_variant == "poly_bstar_cslog_loggauss_odd":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4, alpha;\n  float logx0, sigx, amp;\n  float BNP, c0, c1;\n"
        body = """
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

  float lx = log(x) - logx0;
  float invsig = 1.f / fmax(sigx, 1e-4f);
  float u = lx * invsig;
  float bump = amp * u * expf(-0.5f * u * u);
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
"""
    else:
        return _render_kernel(spec)

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
    base = _best(FAMILY_RESULTS_DIR / "art23_mu_poly_bstar_cslog.json")
    lam1, lam2, lam3, lam4, alpha, BNP, c0, c1 = map(float, base)

    return [
        CandidateSpec(
            name="poly_bstar_cslog_loggauss",
            fit_name="PolyBstarCSLogLogGauss",
            np_name="NP-PolyBstarCSLogLogGauss.cl",
            param_names=["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"],
            initial_params=[lam1, lam2, lam3, lam4, alpha, np.log(0.02), 0.7, 0.0, BNP, c0, c1],
            bounds=[(0.02, 8.0), (0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (np.log(1e-4), np.log(0.3)), (0.15, 2.5), (-2.0, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="poly_bstar_cslog_loggauss",
        ),
        CandidateSpec(
            name="poly_bstar_cslog_loggauss_odd",
            fit_name="PolyBstarCSLogLogGaussOdd",
            np_name="NP-PolyBstarCSLogLogGaussOdd.cl",
            param_names=["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"],
            initial_params=[lam1, lam2, lam3, lam4, alpha, np.log(0.02), 0.7, 0.0, BNP, c0, c1],
            bounds=[(0.02, 8.0), (0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (np.log(1e-4), np.log(0.3)), (0.15, 2.5), (-2.0, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="poly_bstar_cslog_loggauss_odd",
        ),
    ]


def write_followup_candidate_files() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in followup_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
        (NP_DIR / spec.np_name).write_text(_render_kernel_followup(spec), encoding="utf-8")
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
    ref = json.loads((FAMILY_RESULTS_DIR / "art23_mu_poly_bstar_cslog.json").read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]
    rows = []
    for name in specs:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["best"]["metrics"]
        rows.append(
            {
                "candidate": name,
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2dN_collider": metrics["chi2dN_collider"],
                "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "highE_mean_signed_first3": metrics["highE_mean_signed_first3"],
                "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
                "delta_total_vs_poly_bstar_cslog": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
                "delta_absdev_vs_poly_bstar_cslog": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
                "delta_shortfall_vs_poly_bstar_cslog": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
                "fit_evals": result["fit"]["nf"],
                "fit_flag": result["fit"]["flag"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["highE_mean_absdev_first3", "highE_mean_shortfall_first3", "chi2dN_total"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df


def orchestrate(candidates: list[str], maxfun: int) -> None:
    specs = write_followup_candidate_files()
    for name in candidates:
        cmd = [str(PYTHON), str(Path(__file__)), "--candidate", name, "--maxfun", str(maxfun)]
        print(f"\n=== Running {name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{name}: exit code {proc.returncode}")

    print("\n=== Summary ===")
    df = summarize_results(specs)
    if not df.empty:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, help="Run exactly one candidate.")
    parser.add_argument("--maxfun", type=int, default=960)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    write_auto_candidate_files()
    write_family_candidate_files()
    write_refine_candidate_files()
    specs = write_followup_candidate_files()

    if args.candidate:
        out_path = run_candidate(specs[args.candidate], maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    orchestrate(candidates=args.candidates, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
