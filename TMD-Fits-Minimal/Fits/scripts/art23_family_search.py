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
import pybobyqa

from auto_np_search import (
    BASELINE_FULL_PARAMS,
    CARDS_DIR,
    FITS_DIR,
    NP_DIR,
    PYTHON,
    CandidateSpec,
    FitSession,
    candidate_specs as auto_candidate_specs,
    write_candidate_files as write_auto_candidate_files,
)
from _paths import RESULTS_ROOT


RESULTS_DIR = RESULTS_ROOT / "art23_family_results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_CANDIDATES = [
    "baseline_unfrozen",
    "art23_fi_cslog_refit",
    "hybrid_0112_art23cs_refit",
    "art23_mu_poly_cslog",
    "art23_mu_poly_bstar_cslog",
    "art17m1_art23cslog",
    "art17m2_art23cslog",
    "art23_fi_cslog2",
    "art23_mu_poly_cslog2",
]


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

const table_name = "MSHT20N3LO-MC"
const pdf_name = "approximate"
const error_sets_name = "MSHT20N3LO-MC"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"
"""


def _render_kernel(spec: CandidateSpec) -> str:
    variant = spec.kernel_variant

    if variant == "art23_fi_cslog":
        struct_fields = "  float lambda1, lambda2, BNP, c0, c1;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float shape = lambda1 * xbar + lambda2 * x;
  float SNP_mu = sechf(shape * b);

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "art23_fi_cslog2":
        struct_fields = "  float lambda1, lambda2, BNP, c0, c1, c2;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;
  const float c2 = p->c2;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float shape = lambda1 * xbar + lambda2 * x;
  float SNP_mu = sechf(shape * b);

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
    elif variant == "art23_mu_poly_bstar_cslog":
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
    elif variant == "art23_mu_poly_cslog2":
        struct_fields = "  float lambda1, lambda2, lambda3, lambda4;\n  float BNP, c0, c1, c2;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float lambda3 = p->lambda3;
  const float lambda4 = p->lambda4;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;
  const float c2 = p->c2;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;
  float shape = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x);
  float SNP_mu = sechf(shape * b);

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio + c2 * log_ratio * log_ratio);
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
    elif variant == "art17m2_art23cslog2":
        struct_fields = "  float lambda1, lambda2, BNP, c0, c1, c2;\n"
        body = """
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;
  const float c2 = p->c2;

  x = clampf(x, 1e-7f, 1.f);
  float b2 = b * b;
  float ratio = (lambda2 * lambda2) / fmax(lambda1 * lambda1, 1e-12f);
  float denom = sqrtf(1.f + x * x * b2 * ratio);
  float SNP_mu = expf(-lambda2 * x * b2 / fmax(denom, 1e-12f));

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio + c2 * log_ratio * log_ratio);
  float SNP_ze = -DNP;
"""
    elif variant == "art23_mu_poly_bstar_cslog2":
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
    elif variant == "hybrid_0112_art23cs":
        struct_fields = "  float a1, a2, a3, a4;\n  float b1, b2, b3;\n  float a;\n  float BNP, c0, c1;\n"
        body = """
  const float a1 = p->a1, a2 = p->a2, a3 = p->a3, a4 = p->a4;
  const float b1 = p->b1, b2p = p->b2, b3 = p->b3;
  const float alpha = p->a;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float xxbar = x * xbar;

  float xshape = a1 * x + a2 * xbar + a3 * xxbar + a4 * log(x);

  float expo = b1 * x * x + b2p * xbar * xbar + 2.f * b3 * xxbar;
  expo = clampf(expo, -80.f, 80.f);
  float bshape = expf(expo);

  float t = b / (bmax * fmax(bshape, 1e-8f));
  float t2 = t * t;
  float t4 = t2 * t2;
  float bstar_mu = b * powr(1.f + t4, 0.25f * (alpha - 1.f));
  float SNP_mu = sechf(xshape * bstar_mu);

  float v = b / fmax(BNP, 1e-6f);
  float bstar_cs = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar_cs / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar_cs * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;
"""
    else:
        raise ValueError(f"Unknown ART23-family kernel variant: {variant}")

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


def family_specs() -> list[CandidateSpec]:
    art23_init = [0.4530263682496283, 0.8045073983132688, 1.4500086827422805, 0.07419699259000759, 0.021507110056008586]
    hybrid_init = [
        2.452588537440551,
        0.7218607139481996,
        -0.05288953469085733,
        0.03362010183835051,
        -3.866467487924714,
        -1.0419915496345595,
        -1.7862675884664707,
        0.671594894553983,
        1.5928711556172432,
        0.08234438657832673,
        0.033855500067208326,
    ]
    art17m1_init = [0.09535798263172778, 0.09745060199812608, 1.4500086827422805, 0.07419699259000759, 0.021507110056008586]
    art17m2_init = [0.07813285426341661, 1.5017464623554853, 1.4500086827422805, 0.07419699259000759, 0.021507110056008586]

    return [
        CandidateSpec(
            name="art23_fi_cslog_refit",
            fit_name="Art23FamilyFICSLog",
            np_name="NP-Art23FamilyFICSLog.cl",
            param_names=["lambda1", "lambda2", "BNP", "c0", "c1"],
            initial_params=art23_init,
            bounds=[(0.02, 8.0), (0.02, 8.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art23_fi_cslog",
        ),
        CandidateSpec(
            name="art23_fi_cslog2",
            fit_name="Art23FamilyFICSLog2",
            np_name="NP-Art23FamilyFICSLog2.cl",
            param_names=["lambda1", "lambda2", "BNP", "c0", "c1", "c2"],
            initial_params=[*art23_init, 0.0],
            bounds=[(0.02, 8.0), (0.02, 8.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art23_fi_cslog2",
        ),
        CandidateSpec(
            name="art23_mu_poly_cslog",
            fit_name="Art23FamilyMuPolyCSLog",
            np_name="NP-Art23FamilyMuPolyCSLog.cl",
            param_names=["lambda1", "lambda2", "lambda3", "lambda4", "BNP", "c0", "c1"],
            initial_params=[art23_init[0], art23_init[1], 0.0, 0.0, art23_init[2], art23_init[3], art23_init[4]],
            bounds=[(0.02, 8.0), (0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art23_mu_poly_cslog",
        ),
        CandidateSpec(
            name="art23_mu_poly_bstar_cslog",
            fit_name="Art23FamilyMuPolyBstarCSLog",
            np_name="NP-Art23FamilyMuPolyBstarCSLog.cl",
            param_names=["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "BNP", "c0", "c1"],
            initial_params=[art23_init[0], art23_init[1], 0.0, 0.0, 1.0, art23_init[2], art23_init[3], art23_init[4]],
            bounds=[(0.02, 8.0), (0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.0, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art23_mu_poly_bstar_cslog",
        ),
        CandidateSpec(
            name="art23_mu_poly_cslog2",
            fit_name="Art23FamilyMuPolyCSLog2",
            np_name="NP-Art23FamilyMuPolyCSLog2.cl",
            param_names=["lambda1", "lambda2", "lambda3", "lambda4", "BNP", "c0", "c1", "c2"],
            initial_params=[art23_init[0], art23_init[1], 0.0, 0.0, art23_init[2], art23_init[3], art23_init[4], 0.0],
            bounds=[(0.02, 8.0), (0.02, 8.0), (-10.0, 10.0), (-0.5, 0.5), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art23_mu_poly_cslog2",
        ),
        CandidateSpec(
            name="art17m1_art23cslog",
            fit_name="Art23FamilyART17M1CSLog",
            np_name="NP-Art23FamilyART17M1CSLog.cl",
            param_names=["lambda1", "lambda2", "BNP", "c0", "c1"],
            initial_params=art17m1_init,
            bounds=[(0.02, 1.5), (0.01, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art17m1_art23cslog",
        ),
        CandidateSpec(
            name="art17m2_art23cslog",
            fit_name="Art23FamilyART17M2CSLog",
            np_name="NP-Art23FamilyART17M2CSLog.cl",
            param_names=["lambda1", "lambda2", "BNP", "c0", "c1"],
            initial_params=art17m2_init,
            bounds=[(0.02, 1.5), (0.01, 2.0), (0.4, 4.5), (0.0, 0.25), (0.0, 0.25)],
            frozen_indices=[],
            kernel_variant="art17m2_art23cslog",
        ),
        CandidateSpec(
            name="hybrid_0112_art23cs_refit",
            fit_name="Art23FamilyHybrid0112ART23CS",
            np_name="NP-Art23FamilyHybrid0112ART23CS.cl",
            param_names=["a1", "a2", "a3", "a4", "b1", "b2", "b3", "a", "BNP", "c0", "c1"],
            initial_params=hybrid_init,
            bounds=[
                (-10.0, 10.0),
                (-10.0, 10.0),
                (-10.0, 10.0),
                (-0.5, 0.5),
                (-10.0, 10.0),
                (-10.0, 10.0),
                (-10.0, 10.0),
                (0.0, 2.0),
                (0.4, 4.5),
                (0.0, 0.25),
                (0.0, 0.25),
            ],
            frozen_indices=[],
            kernel_variant="hybrid_0112_art23cs",
        ),
    ]


def write_family_candidate_files() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in family_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
        (NP_DIR / spec.np_name).write_text(_render_kernel(spec), encoding="utf-8")
    return specs


class Art23FamilyFitSession(FitSession):
    def candidate_start_points(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = [self.theta0.copy()]
        seed = 20260322 + sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for scale in [0.03, 0.06, 0.10, 0.14, 0.20]:
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))

        anchor_values = {
            "lambda1": [0.06, 0.20, 0.45, 0.75],
            "lambda2": [0.06, 0.20, 0.45, 0.75],
            "lambda3": [0.10, 0.50, 0.90],
            "lambda4": [0.10, 0.50, 0.90],
            "alpha": [0.10, 0.40, 0.70, 0.90],
            "a1": [0.10, 0.50, 0.90],
            "a2": [0.10, 0.50, 0.90],
            "a3": [0.10, 0.50, 0.90],
            "a4": [0.10, 0.50, 0.90],
            "a": [0.10, 0.50, 0.90],
            "b1": [0.10, 0.50, 0.90],
            "b2": [0.10, 0.50, 0.90],
            "b3": [0.10, 0.50, 0.90],
            "BNP": [0.10, 0.35, 0.60, 0.85],
            "c0": [0.05, 0.20, 0.40, 0.70],
            "c1": [0.05, 0.20, 0.40, 0.70],
            "c2": [0.05, 0.20, 0.40, 0.70],
        }

        for pname, vals in anchor_values.items():
            if pname not in self.free_param_names:
                continue
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
        for item in stage1_results[:4]:
            res = self._solve(item["theta"], maxfun=stage2_budget, rhobeg=0.10, rhoend=1e-4, seek_global_minimum=True)
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
        for item in stage2_results[:2]:
            res = self._solve(item["theta"], maxfun=refine_budget, rhobeg=0.05, rhoend=2e-5, seek_global_minimum=False)
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
        res = self._solve(best_refined["theta"], maxfun=polish_budget, rhobeg=0.025, rhoend=1e-6, seek_global_minimum=False)

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


def all_specs() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in auto_candidate_specs() if spec.name == "baseline_unfrozen"}
    specs.update(write_family_candidate_files())
    return specs


def run_candidate(spec: CandidateSpec, maxfun: int) -> Path:
    session_cls = Art23FamilyFitSession if spec.name != "baseline_unfrozen" else Art23FamilyFitSession
    session = session_cls(spec)
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
    ref_name = "baseline_unfrozen"
    ref = json.loads((RESULTS_DIR / f"{ref_name}.json").read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]

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
                "highE_weighted_chi2dN": metrics["highE_weighted_chi2dN"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "highE_mean_signed_first3": metrics["highE_mean_signed_first3"],
                "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
                "delta_total_vs_ref": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
                "delta_absdev_vs_ref": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
                "delta_shortfall_vs_ref": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
                "fit_evals": result["fit"]["nf"],
                "fit_flag": result["fit"]["flag"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    if not df.empty:
        top_chi2_name = str(df.iloc[0]["candidate"])
        pd.DataFrame(json.loads((RESULTS_DIR / f"{top_chi2_name}.json").read_text(encoding="utf-8"))["best"]["high_energy_rows"]).to_csv(
            RESULTS_DIR / f"{top_chi2_name}_high_energy_rows.csv", index=False
        )
        top_absdev_name = str(df.sort_values(["highE_mean_absdev_first3", "chi2dN_total"]).iloc[0]["candidate"])
        pd.DataFrame(json.loads((RESULTS_DIR / f"{top_absdev_name}.json").read_text(encoding="utf-8"))["best"]["high_energy_rows"]).to_csv(
            RESULTS_DIR / f"{top_absdev_name}_high_energy_rows.csv", index=False
        )
    return df


def orchestrate(candidates: list[str], maxfun: int) -> None:
    specs = all_specs()
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
    parser.add_argument("--maxfun", type=int, default=320)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    write_auto_candidate_files()
    specs = all_specs()

    if args.candidate:
        out_path = run_candidate(specs[args.candidate], maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    orchestrate(candidates=args.candidates, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
