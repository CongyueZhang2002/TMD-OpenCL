from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pybobyqa

from _paths import CARDS_DIR, FITS_DIR, NP_DIR, RESULTS_ROOT, ROOT

RESULTS_DIR = RESULTS_ROOT / "auto_np_results_round2"
RESULTS_DIR.mkdir(exist_ok=True)

PYTHON = Path(sys.executable)
JULIA_RUNTIME = Path(r"C:\Users\congyue zhang\AppData\Local\Programs\Julia-1.11.6\bin\julia.exe")


EXPERIMENTS = [
    "ATLAS_7",
    "ATLAS_8",
    "CDF_I",
    "CDF_II",
    "CMS_7",
    "CMS_8",
    "CMS_13",
    "D0_I",
    "D0_II",
    "D0_II_mu",
    "E288",
    "E605",
    "E772",
    "LHCb_7",
    "LHCb_8",
    "LHCb_13",
    "STAR",
]

FILE_EXCLUDES = {
    "E772/E772-5Q6.csv",
    "E772/E772-6Q7.csv",
    "E772/E772-7Q8.csv",
    "E772/E772-8Q9.csv",
}

COLLIDER_PREFIXES = ("ATLAS_", "CMS_", "LHCb_", "D0_", "CDF_", "STAR")
FIXED_TARGET_PREFIXES = ("E288\\", "E605\\", "E772\\")

HIGH_ENERGY_FILES = [
    r"CMS_13\CMS13-170Q350.csv",
    r"CMS_13\CMS13-350Q1000.csv",
    r"ATLAS_8\ATLAS8-00y04.csv",
    r"ATLAS_8\ATLAS8-04y08.csv",
    r"LHCb_8\LHCb8.csv",
]

BASELINE_FULL_PARAMS = [
    0.50903666,
    1.2073716,
    0.9093148,
    2.8733068,
    -0.0066525186,
    0.0,
    0.0,
    -3.0842779,
    0.15909232,
    0.0,
    0.0,
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    fit_name: str
    np_name: str
    param_names: list[str]
    initial_params: list[float]
    bounds: list[tuple[float, float]]
    frozen_indices: list[int]
    kernel_variant: str


def _norm_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _render_np_kernel(spec: CandidateSpec) -> str:
    extra_struct = ""
    extra_body = ""
    mu_expr = "sechf(xshape * bstar)"
    cs_coeff_expr = "0.25f * (g2*g2)"

    if spec.kernel_variant == "baseline":
        pass
    elif spec.kernel_variant == "xcs_linear":
        extra_struct = ", cx"
        cs_coeff_expr = "clampf(0.25f * (g2*g2) + cx * logxrel, -2.f, 4.f)"
    elif spec.kernel_variant == "xcs_quad":
        extra_struct = ", cx1, cx2"
        cs_coeff_expr = "clampf(0.25f * (g2*g2) + cx1 * logxrel + cx2 * logxrel * logxrel, -2.f, 4.f)"
    elif spec.kernel_variant == "xwindow_combo":
        extra_struct = ", cx, wx, bm"
        extra_body = (
            "  float v = b / bm;\n"
            "  float h_b = (b*b) / (1.f + v*v);\n"
        )
        cs_coeff_expr = "clampf(0.25f * (g2*g2) + cx * logxrel, -2.f, 4.f)"
    elif spec.kernel_variant == "xmu_xcs_combo":
        extra_struct = ", cx, mx, bm"
        extra_body = (
            "  float v = b / bm;\n"
            "  float h_b = (b*b) / (1.f + v*v);\n"
        )
        mu_expr = "sechf(xshape * bstar) * exp(clampf(mx * logxrel * h_b, -1.5f, 1.5f))"
        cs_coeff_expr = "clampf(0.25f * (g2*g2) + cx * logxrel, -2.f, 4.f)"
    else:
        raise ValueError(f"Unknown kernel variant: {spec.kernel_variant}")

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
  float g2, bmax_CS, power_CS{extra_struct};
  float a1,a2,a3,a4;
  float b1,b2,b3;
  float a;
}} Params_Struct;

inline float clampf(float x,float lo,float hi){{ return fmin(fmax(x,lo),hi); }}
inline float sechf(float t){{ t=fabs(t); float u=exp(-2.f*t); return (2.f*exp(-t))/(1.f+u); }}

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{{
  const float g2       = p->g2;
  const float bmax_CS  = p->bmax_CS;
  const float power_CS = p->power_CS;
{_extra_param_loads(spec)}
  const float a1 = p->a1, a2 = p->a2, a3 = p->a3, a4 = p->a4;
  const float b1 = p->b1, b2 = p->b2, b3 = p->b3;
  const float alpha = p->a;

  x = clampf(x, 1e-7f, 1.f-1e-7f);
  float xbar = 1.f - x, xxbar = x*xbar;

  float xshape = a1*x + a2*xbar + a3*xxbar + a4*log(x);

  float expo = b1*x*x + b2*xbar*xbar + 2.f*b3*xxbar;
  expo = clampf(expo, -80.f, 80.f);
  float bshape = exp(expo);

  float t  = b/(bmax*bshape);
  float t2 = t*t;
  float t4 = t2*t2;
  float bstar = b * powr(1.f + t4, 0.25f*(alpha - 1.f));

  float u = b/bmax_CS;
  float u2 = u*u;
  float u4 = u2*u2;
  float bstar_CS = b * powr(1.f + u4, 0.25f*(power_CS - 1.f));

  float logxrel = log(x / xh);
  float Lx = fmax(0.f, -logxrel);
{extra_body}  float cs_coeff = {cs_coeff_expr};
  float SNP_mu = {mu_expr};
  float SNP_ze = -cs_coeff * (bstar_CS*bstar_CS);
{_extra_sudakov_tail(spec)}

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


def _extra_param_loads(spec: CandidateSpec) -> str:
    if spec.kernel_variant == "baseline":
        return ""
    if spec.kernel_variant == "xcs_linear":
        return "  const float cx       = p->cx;\n"
    if spec.kernel_variant == "xcs_quad":
        return "  const float cx1      = p->cx1;\n  const float cx2      = p->cx2;\n"
    if spec.kernel_variant == "xwindow_combo":
        return (
            "  const float cx       = p->cx;\n"
            "  const float wx       = p->wx;\n"
            "  const float bm       = p->bm;\n"
        )
    if spec.kernel_variant == "xmu_xcs_combo":
        return (
            "  const float cx       = p->cx;\n"
            "  const float mx       = p->mx;\n"
            "  const float bm       = p->bm;\n"
        )
    raise ValueError(spec.kernel_variant)


def _extra_sudakov_tail(spec: CandidateSpec) -> str:
    if spec.kernel_variant == "xwindow_combo":
        return "  SNP_ze += -wx * logxrel * h_b;\n"
    return ""


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


def candidate_specs() -> list[CandidateSpec]:
    baseline_bounds = [
        (0.0, 1.0),
        (0.5, 5.0),
        (0.0, 2.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-0.5, 0.5),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (0.0, 2.0),
    ]

    baseline_names = ["g2", "bmax_CS", "power_CS", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "a"]

    baseline_unfrozen_initial = BASELINE_FULL_PARAMS.copy()

    xlin_names = ["g2", "bmax_CS", "power_CS", "cx", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "a"]
    xlin_initial = [BASELINE_FULL_PARAMS[0], BASELINE_FULL_PARAMS[1], BASELINE_FULL_PARAMS[2], 0.0,
                    BASELINE_FULL_PARAMS[3], BASELINE_FULL_PARAMS[4], BASELINE_FULL_PARAMS[5], BASELINE_FULL_PARAMS[6],
                    BASELINE_FULL_PARAMS[7], BASELINE_FULL_PARAMS[8], BASELINE_FULL_PARAMS[9], BASELINE_FULL_PARAMS[10]]
    xlin_bounds = [
        (0.0, 1.0), (0.5, 5.0), (0.0, 2.0), (-0.5, 0.5),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-0.5, 0.5),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, 2.0),
    ]

    xquad_names = ["g2", "bmax_CS", "power_CS", "cx1", "cx2", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "a"]
    xquad_initial = [BASELINE_FULL_PARAMS[0], BASELINE_FULL_PARAMS[1], BASELINE_FULL_PARAMS[2], 0.0, 0.0,
                     BASELINE_FULL_PARAMS[3], BASELINE_FULL_PARAMS[4], BASELINE_FULL_PARAMS[5], BASELINE_FULL_PARAMS[6],
                     BASELINE_FULL_PARAMS[7], BASELINE_FULL_PARAMS[8], BASELINE_FULL_PARAMS[9], BASELINE_FULL_PARAMS[10]]
    xquad_bounds = [
        (0.0, 1.0), (0.5, 5.0), (0.0, 2.0), (-0.6, 0.6), (-0.3, 0.3),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-0.5, 0.5),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, 2.0),
    ]

    xwindow_names = ["g2", "bmax_CS", "power_CS", "cx", "wx", "bm", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "a"]
    xwindow_initial = [BASELINE_FULL_PARAMS[0], BASELINE_FULL_PARAMS[1], BASELINE_FULL_PARAMS[2], 0.0, 0.0, 1.0,
                       BASELINE_FULL_PARAMS[3], BASELINE_FULL_PARAMS[4], BASELINE_FULL_PARAMS[5], BASELINE_FULL_PARAMS[6],
                       BASELINE_FULL_PARAMS[7], BASELINE_FULL_PARAMS[8], BASELINE_FULL_PARAMS[9], BASELINE_FULL_PARAMS[10]]
    xwindow_bounds = [
        (0.0, 1.0), (0.5, 5.0), (0.0, 2.0), (-0.5, 0.5), (-0.5, 0.5), (0.2, 5.0),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-0.5, 0.5),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, 2.0),
    ]

    xmu_names = ["g2", "bmax_CS", "power_CS", "cx", "mx", "bm", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "a"]
    xmu_initial = [BASELINE_FULL_PARAMS[0], BASELINE_FULL_PARAMS[1], BASELINE_FULL_PARAMS[2], 0.0, 0.0, 1.0,
                   BASELINE_FULL_PARAMS[3], BASELINE_FULL_PARAMS[4], BASELINE_FULL_PARAMS[5], BASELINE_FULL_PARAMS[6],
                   BASELINE_FULL_PARAMS[7], BASELINE_FULL_PARAMS[8], BASELINE_FULL_PARAMS[9], BASELINE_FULL_PARAMS[10]]
    xmu_bounds = [
        (0.0, 1.0), (0.5, 5.0), (0.0, 2.0), (-0.5, 0.5), (-0.2, 0.2), (0.2, 5.0),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-0.5, 0.5),
        (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, 2.0),
    ]

    return [
        CandidateSpec(
            name="baseline_0112",
            fit_name="AutoBaseline0112",
            np_name="NP-AutoBaseline0112.cl",
            param_names=baseline_names,
            initial_params=BASELINE_FULL_PARAMS,
            bounds=baseline_bounds,
            frozen_indices=[5, 6, 9, 10],
            kernel_variant="baseline",
        ),
        CandidateSpec(
            name="baseline_unfrozen",
            fit_name="AutoBaselineUnfrozen",
            np_name="NP-AutoBaselineUnfrozen.cl",
            param_names=baseline_names,
            initial_params=baseline_unfrozen_initial,
            bounds=baseline_bounds,
            frozen_indices=[],
            kernel_variant="baseline",
        ),
        CandidateSpec(
            name="xcs_linear",
            fit_name="AutoXCSLinear",
            np_name="NP-AutoXCSLinear.cl",
            param_names=xlin_names,
            initial_params=xlin_initial,
            bounds=xlin_bounds,
            frozen_indices=[6, 7, 10, 11],
            kernel_variant="xcs_linear",
        ),
        CandidateSpec(
            name="xcs_quad",
            fit_name="AutoXCSQuad",
            np_name="NP-AutoXCSQuad.cl",
            param_names=xquad_names,
            initial_params=xquad_initial,
            bounds=xquad_bounds,
            frozen_indices=[7, 8, 11, 12],
            kernel_variant="xcs_quad",
        ),
        CandidateSpec(
            name="xwindow_combo",
            fit_name="AutoXWindowCombo",
            np_name="NP-AutoXWindowCombo.cl",
            param_names=xwindow_names,
            initial_params=xwindow_initial,
            bounds=xwindow_bounds,
            frozen_indices=[8, 9, 12, 13],
            kernel_variant="xwindow_combo",
        ),
        CandidateSpec(
            name="xmu_xcs_combo",
            fit_name="AutoXMuXCSCombo",
            np_name="NP-AutoXMuXCSCombo.cl",
            param_names=xmu_names,
            initial_params=xmu_initial,
            bounds=xmu_bounds,
            frozen_indices=[8, 9, 12, 13],
            kernel_variant="xmu_xcs_combo",
        ),
    ]


def write_candidate_files() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in candidate_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
        (NP_DIR / spec.np_name).write_text(_render_np_kernel(spec), encoding="utf-8")
    return specs


class FitSession:
    def __init__(self, spec: CandidateSpec) -> None:
        from julia.api import Julia

        self.jl = Julia(runtime=str(JULIA_RUNTIME), compiled_modules=False)
        from julia import Main

        self.Main = Main
        self.spec = spec
        self.fit_name = spec.fit_name
        self._include(str(CARDS_DIR / f"{self.fit_name}.jl"))
        self._include(str(ROOT / "DY" / f"DY_table_{self.Main.flavor_scheme}.jl"))

        self.approximate_total_xsec = True
        self.data_uncertainty_only = False

        self.file_root = ROOT / "Data" / self.Main.data_name / "Cutted" / "DY"
        self.matrix_root = ROOT / "Data" / self.Main.data_name / "Covariance_Matrices" / "DY"
        self.table_root = ROOT / "Tables" / self.Main.table_name / "DY"
        self.table_root_rel = Path("..") / "Tables" / self.Main.table_name / "DY"
        self.total_root = ROOT / "Data" / "DY_total_xsec" / self.Main.pdf_name
        self.error_sets_root = ROOT / "Data" / "PDF_Matrices" / self.Main.error_sets_name / "DY"

        self.initial_params = np.asarray(self.Main.initial_params, dtype=float)
        self.frozen_idx = np.asarray(self.Main.frozen_indices, dtype=int)
        mask = np.ones(len(self.initial_params), dtype=bool)
        mask[self.frozen_idx] = False
        self.free_idx = np.where(mask)[0]
        self.frozen_vals = self.initial_params[self.frozen_idx].copy()

        self.bounds_full = np.asarray(self.Main.bounds_raw, dtype=float)
        self.bounds_free = self.bounds_full[self.free_idx]
        self.lower_bounds = self.bounds_free[:, 0]
        self.upper_bounds = self.bounds_free[:, 1]
        self.theta0 = self.normalize_params(self.initial_params[self.free_idx])
        self.free_param_names = [self.spec.param_names[i] for i in self.free_idx]

        self.file_names = self._build_file_list()
        self.file_lengths = self._get_file_lengths()
        self.data_list, self.matrix_inv_list, self.n_list = self._load_data()
        self.total_xsec_names = set(pd.read_csv(self.total_root.with_suffix(".csv"))["name"].tolist())

        # Warm up the OpenCL path once.
        self._predict(self.initial_params)

    def _include(self, path: str) -> None:
        self.Main.eval(f'include(raw"{path}")')

    def _build_file_list(self) -> list[str]:
        file_names: list[str] = []
        for experiment in EXPERIMENTS:
            exp_dir = self.file_root / experiment
            for path in sorted(exp_dir.glob("*.csv")):
                rel = str(Path(experiment) / path.name)
                if rel.replace("\\", "/") in FILE_EXCLUDES:
                    continue
                file_names.append(rel)
        return file_names

    def _get_file_lengths(self) -> dict[str, int]:
        lengths: dict[str, int] = {}
        for file_name in self.file_names:
            df = pd.read_csv(self.file_root / file_name)
            lengths[file_name] = len(df)
        return lengths

    def _load_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, int]]:
        data_list: dict[str, pd.DataFrame] = {}
        matrix_inv_list: dict[str, np.ndarray] = {}
        n_list: dict[str, int] = {}

        df_total_xsec = pd.read_csv(self.total_root.with_suffix(".csv"))
        total_xsec_lookup = {
            row["name"]: float(row["total_xsec"])
            for _, row in df_total_xsec.iterrows()
        }

        for file_name in self.file_names:
            df_data = pd.read_csv(self.file_root / file_name)
            data_list[file_name] = df_data

            matrix_data = pd.read_csv(self.matrix_root / file_name).to_numpy(dtype=float)
            if self.data_uncertainty_only:
                matrix_total = matrix_data
            else:
                matrix_pdf = pd.read_csv(self.error_sets_root / file_name).to_numpy(dtype=float)
                matrix_total = matrix_data + matrix_pdf

            matrix_inv_list[file_name] = np.linalg.inv(matrix_total)
            n_list[file_name] = len(df_data)

            stem = Path(file_name).stem
            if stem in total_xsec_lookup:
                data_list[file_name]["total_xsec"] = total_xsec_lookup[stem] * np.ones(len(df_data))

        return data_list, matrix_inv_list, n_list

    def normalize_params(self, params_free: np.ndarray) -> np.ndarray:
        return (params_free - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def denormalize_params(self, theta: np.ndarray) -> np.ndarray:
        return self.lower_bounds + theta * (self.upper_bounds - self.lower_bounds)

    def full_from_free(self, params_free: np.ndarray) -> np.ndarray:
        full = self.initial_params.copy()
        full[self.free_idx] = np.asarray(params_free, dtype=float)
        full[self.frozen_idx] = self.frozen_vals
        return full

    def _predict(self, params_full: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        params_cl = self.Main.Params_Struct(*[np.float32(x) for x in params_full])
        self.Main.set_params(self.Main.VRAM, params_cl)
        predictions, compute_s = self.Main.xsec_dict(self.Main.rel_paths, self.Main.VRAM)
        return self._prediction_reformat(predictions), float(compute_s)

    def _prediction_reformat(self, predictions) -> dict[str, np.ndarray]:
        preds = {_norm_path(k): float(v) for k, v in predictions.items()}
        df_predictions: dict[str, np.ndarray] = {}

        for file_name in self.file_names:
            n = self.file_lengths[file_name]
            base = os.path.splitext(file_name)[0]
            xs = []
            for i in range(n):
                table_path = _norm_path(os.path.join(str(self.table_root_rel), f"{base}/{i}.jls"))
                xs.append(preds[table_path])
            arr = np.asarray(xs, dtype=float)

            if self.approximate_total_xsec and Path(file_name).stem in self.total_xsec_names:
                data_xsec = self.data_list[file_name]["xsec"].to_numpy(dtype=float)
                scale = float(np.sum(data_xsec) / np.sum(arr))
                arr = scale * arr

            df_predictions[file_name] = arr

        return df_predictions

    def get_chi2(self, predictions: dict[str, np.ndarray]) -> tuple[float, dict[str, float], dict[str, int]]:
        chi2_total = 0.0
        n_total = 0
        chi2_per_file: dict[str, float] = {}

        for file_name in self.file_names:
            data_xsec = self.data_list[file_name]["xsec"].to_numpy(dtype=float)
            pred_xsec = predictions[file_name]
            diff = data_xsec - pred_xsec
            chi2 = float(diff @ self.matrix_inv_list[file_name] @ diff)
            n = self.n_list[file_name]
            chi2_total += chi2
            n_total += n
            chi2_per_file[file_name] = chi2 / n

        return chi2_total / n_total, chi2_per_file, self.n_list

    def weighted_subset_chi2(self, chi2_list: dict[str, float], prefixes: tuple[str, ...]) -> float:
        names = [name for name in chi2_list if name.startswith(prefixes)]
        if not names:
            return math.nan
        total_n = sum(self.n_list[name] for name in names)
        total_chi2 = sum(chi2_list[name] * self.n_list[name] for name in names)
        return total_chi2 / total_n

    def weighted_exact_chi2(self, chi2_list: dict[str, float], names: list[str]) -> float:
        present = [name for name in names if name in chi2_list]
        if not present:
            return math.nan
        total_n = sum(self.n_list[name] for name in present)
        total_chi2 = sum(chi2_list[name] * self.n_list[name] for name in present)
        return total_chi2 / total_n

    def high_energy_rows(self, predictions: dict[str, np.ndarray], chi2_list: dict[str, float]) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for file_name in HIGH_ENERGY_FILES:
            data = self.data_list[file_name]["xsec"].to_numpy(dtype=float)
            pred = predictions[file_name]
            row: dict[str, float | str] = {
                "file": file_name,
                "chi2dN": chi2_list[file_name],
                "N": self.n_list[file_name],
            }
            for i in range(min(3, len(data))):
                ratio = float(pred[i] / data[i])
                row[f"ratio_{i + 1}"] = ratio
                row[f"data_{i + 1}"] = float(data[i])
                row[f"pred_{i + 1}"] = float(pred[i])
            rows.append(row)
        return rows

    def summary_metrics(self, predictions: dict[str, np.ndarray], chi2_list: dict[str, float], compute_s: float) -> dict[str, float]:
        total_chi2dN, _, _ = self.get_chi2(predictions)
        ratios = []
        absdev = []
        signed = []
        shortfall = []
        overshoot = []
        cms_ratios = []
        zlike_ratios = []
        for file_name in HIGH_ENERGY_FILES:
            data = self.data_list[file_name]["xsec"].to_numpy(dtype=float)
            pred = predictions[file_name]
            for i in range(min(3, len(data))):
                ratio = float(pred[i] / data[i])
                ratios.append(ratio)
                absdev.append(abs(ratio - 1.0))
                signed.append(ratio - 1.0)
                shortfall.append(max(0.0, 1.0 - ratio))
                overshoot.append(max(0.0, ratio - 1.0))
                if file_name.startswith(r"CMS_13\CMS13-"):
                    cms_ratios.append(ratio)
                else:
                    zlike_ratios.append(ratio)

        return {
            "chi2dN_total": float(total_chi2dN),
            "chi2dN_collider": float(self.weighted_subset_chi2(chi2_list, COLLIDER_PREFIXES)),
            "chi2dN_fixed_target": float(self.weighted_subset_chi2(chi2_list, FIXED_TARGET_PREFIXES)),
            "highE_weighted_chi2dN": float(self.weighted_exact_chi2(chi2_list, HIGH_ENERGY_FILES)),
            "highE_mean_ratio_first3": float(np.mean(ratios)),
            "highE_mean_absdev_first3": float(np.mean(absdev)),
            "highE_mean_signed_first3": float(np.mean(signed)),
            "highE_mean_shortfall_first3": float(np.mean(shortfall)),
            "highE_mean_overshoot_first3": float(np.mean(overshoot)),
            "cms_highmass_mean_ratio_first3": float(np.mean(cms_ratios)),
            "cms_highmass_mean_absdev_first3": float(np.mean(np.abs(np.asarray(cms_ratios) - 1.0))),
            "cms_highmass_mean_signed_first3": float(np.mean(np.asarray(cms_ratios) - 1.0)),
            "zlike_mean_ratio_first3": float(np.mean(zlike_ratios)),
            "zlike_mean_absdev_first3": float(np.mean(np.abs(np.asarray(zlike_ratios) - 1.0))),
            "zlike_mean_signed_first3": float(np.mean(np.asarray(zlike_ratios) - 1.0)),
            "compute_s": float(compute_s),
        }

    def evaluate_free(self, params_free: np.ndarray) -> dict:
        full = self.full_from_free(params_free)
        predictions, compute_s = self._predict(full)
        chi2dN, chi2_list, _ = self.get_chi2(predictions)
        return {
            "full_params": full.tolist(),
            "free_params": np.asarray(params_free, dtype=float).tolist(),
            "metrics": self.summary_metrics(predictions, chi2_list, compute_s),
            "per_file_chi2dN": {k: float(v) for k, v in chi2_list.items()},
            "high_energy_rows": self.high_energy_rows(predictions, chi2_list),
        }

    def objective_free(self, params_free: np.ndarray) -> float:
        try:
            return float(self.evaluate_free(params_free)["metrics"]["chi2dN_total"])
        except Exception:
            return 1e5

    def objective_log_normalized(self, theta: np.ndarray) -> float:
        params_free = self.denormalize_params(np.asarray(theta, dtype=float))
        value = max(self.objective_free(params_free), 1e-12)
        return float(np.log10(value))

    def candidate_start_points(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = [self.theta0.copy()]
        seed = 20260321 + sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for scale in [0.05, 0.10]:
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))

        for pname in ["cx", "cx1", "cx2", "wx", "mx", "bm", "a4", "a", "a3", "b3"]:
            if pname not in self.free_param_names:
                continue
            idx = self.free_param_names.index(pname)
            for val in [0.2, 0.8]:
                start = self.theta0.copy()
                start[idx] = val
                starts.append(start)

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique[:6]

    def _solve(self, theta0: np.ndarray, maxfun: int, rhobeg: float, rhoend: float, seek_global_minimum: bool) -> pybobyqa.solver.OptimResults:
        return pybobyqa.solve(
            self.objective_log_normalized,
            theta0,
            bounds=(np.zeros_like(self.theta0), np.ones_like(self.theta0)),
            maxfun=maxfun,
            rhobeg=rhobeg,
            rhoend=rhoend,
            scaling_within_bounds=True,
            seek_global_minimum=seek_global_minimum,
        )

    def fit(self, maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1_budget = max(35, maxfun // 3)
        stage2_budget = maxfun
        polish_budget = max(50, maxfun // 2)

        stage1_results = []
        for i, start in enumerate(self.candidate_start_points()):
            res = self._solve(start, maxfun=stage1_budget, rhobeg=0.16, rhoend=5e-4, seek_global_minimum=True)
            stage1_results.append({
                "start_id": i,
                "theta": np.asarray(res.x, dtype=float),
                "log10_chi2": float(res.f),
                "nf": int(res.nf),
                "flag": int(res.flag),
            })

        stage1_results.sort(key=lambda item: item["log10_chi2"])
        stage2_results = []
        for item in stage1_results[:2]:
            res = self._solve(item["theta"], maxfun=stage2_budget, rhobeg=0.09, rhoend=8e-5, seek_global_minimum=True)
            stage2_results.append({
                "theta": np.asarray(res.x, dtype=float),
                "log10_chi2": float(res.f),
                "nf": int(res.nf),
                "flag": int(res.flag),
            })

        stage2_results.sort(key=lambda item: item["log10_chi2"])
        best_stage2 = stage2_results[0]
        res = self._solve(best_stage2["theta"], maxfun=polish_budget, rhobeg=0.04, rhoend=1e-6, seek_global_minimum=False)

        elapsed = time.perf_counter() - t0
        params_free = self.denormalize_params(np.asarray(res.x, dtype=float))
        return {
            "elapsed_s": elapsed,
            "nf": int(sum(item["nf"] for item in stage1_results) + sum(item["nf"] for item in stage2_results) + int(res.nf)),
            "flag": int(res.flag),
            "free_params": params_free.tolist(),
            "log10_chi2": float(res.f),
            "stage1": [
                {
                    "start_id": item["start_id"],
                    "log10_chi2": item["log10_chi2"],
                    "nf": item["nf"],
                    "flag": item["flag"],
                }
                for item in stage1_results
            ],
            "stage2": [
                {
                    "log10_chi2": item["log10_chi2"],
                    "nf": item["nf"],
                    "flag": item["flag"],
                }
                for item in stage2_results
            ],
            "polish_nf": int(res.nf),
        }


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


def summarize_results(specs: dict[str, CandidateSpec]) -> None:
    rows = []
    baseline = json.loads((RESULTS_DIR / "baseline_0112.json").read_text(encoding="utf-8"))
    baseline_metrics = baseline["best"]["metrics"]

    for name in specs:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["best"]["metrics"]
        rows.append({
            "candidate": name,
            "chi2dN_total": metrics["chi2dN_total"],
            "chi2dN_collider": metrics["chi2dN_collider"],
            "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
            "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
            "highE_mean_signed_first3": metrics["highE_mean_signed_first3"],
            "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
            "highE_mean_overshoot_first3": metrics["highE_mean_overshoot_first3"],
            "cms_highmass_mean_absdev_first3": metrics["cms_highmass_mean_absdev_first3"],
            "cms_highmass_mean_signed_first3": metrics["cms_highmass_mean_signed_first3"],
            "zlike_mean_absdev_first3": metrics["zlike_mean_absdev_first3"],
            "zlike_mean_signed_first3": metrics["zlike_mean_signed_first3"],
            "delta_total_vs_baseline": metrics["chi2dN_total"] - baseline_metrics["chi2dN_total"],
            "delta_highE_absdev_vs_baseline": metrics["highE_mean_absdev_first3"] - baseline_metrics["highE_mean_absdev_first3"],
            "delta_highE_shortfall_vs_baseline": metrics["highE_mean_shortfall_first3"] - baseline_metrics["highE_mean_shortfall_first3"],
            "fit_evals": result["fit"]["nf"],
            "fit_flag": result["fit"]["flag"],
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["highE_mean_absdev_first3", "chi2dN_total"])
    summary_csv = RESULTS_DIR / "summary.csv"
    df.to_csv(summary_csv, index=False)
    print(df.to_string(index=False))

    best_name = str(df.iloc[0]["candidate"])
    best_result = json.loads((RESULTS_DIR / f"{best_name}.json").read_text(encoding="utf-8"))
    best_rows = pd.DataFrame(best_result["best"]["high_energy_rows"])
    best_rows.to_csv(RESULTS_DIR / f"{best_name}_high_energy_rows.csv", index=False)
    print("\nBest candidate high-energy rows:")
    print(best_rows.to_string(index=False))


def orchestrate(maxfun: int) -> None:
    specs = write_candidate_files()
    for name in specs:
        cmd = [str(PYTHON), str(Path(__file__)), "--candidate", name, "--maxfun", str(maxfun)]
        print(f"\n=== Running {name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{name}: exit code {proc.returncode}")
    print("\n=== Summary ===")
    summarize_results(specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, help="Run exactly one candidate.")
    parser.add_argument("--maxfun", type=int, default=180)
    args = parser.parse_args()

    specs = write_candidate_files()
    if args.candidate:
        spec = specs[args.candidate]
        out_path = run_candidate(spec, maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(maxfun=args.maxfun)


if __name__ == "__main__":
    main()
