from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pybobyqa

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, CandidateSpec, FitSession
from scan_table_variants import mustar_function_src


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "parameter_significance_0_2_results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_TABLE = "MSHT20N3LO-MC-0-2"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    label: str
    card_path: Path
    kernel_path: Path
    result_path: Path


MODELS = [
    ModelConfig(
        name="baseline_unfrozen",
        label="0112 baseline",
        card_path=CARDS_DIR / "AutoBaselineUnfrozen.jl",
        kernel_path=NP_DIR / "NP-AutoBaselineUnfrozen.cl",
        result_path=FITS_DIR / "power_table_refit_results" / "baseline_unfrozen__MSHT20N3LO-MC-0-2.json",
    ),
    ModelConfig(
        name="poly_bstar_cslog",
        label="Poly-x bstar + CSlog",
        card_path=CARDS_DIR / "Art23FamilyMuPolyBstarCSLog.jl",
        kernel_path=NP_DIR / "NP-Art23FamilyMuPolyBstarCSLog.cl",
        result_path=FITS_DIR / "power_table_refit_results" / "poly_bstar_cslog__MSHT20N3LO-MC-0-2.json",
    ),
    ModelConfig(
        name="poly_bstar_cslog_loggauss",
        label="Poly-x bstar + CSlog + loggauss",
        card_path=CARDS_DIR / "PolyBstarCSLogLogGauss.jl",
        kernel_path=NP_DIR / "NP-PolyBstarCSLogLogGauss.cl",
        result_path=FITS_DIR / "power_table_refit_results" / "poly_bstar_cslog_loggauss__MSHT20N3LO-MC-0-2.json",
    ),
]


REFERENCE_MAP = {
    "baseline_unfrozen": {
        "g2": 0.0,
        "power_CS": 1.0,
        "a1": 0.0,
        "a2": 0.0,
        "a3": 0.0,
        "a4": 0.0,
        "b1": 0.0,
        "b2": 0.0,
        "b3": 0.0,
        "a": 1.0,
    },
    "poly_bstar_cslog": {
        "lambda3": 0.0,
        "lambda4": 0.0,
        "alpha": 1.0,
        "c1": 0.0,
    },
    "poly_bstar_cslog_loggauss": {
        "lambda3": 0.0,
        "lambda4": 0.0,
        "alpha": 1.0,
        "amp": 0.0,
        "c1": 0.0,
    },
}


def null_tests_for(model_name: str, best_full: list[float], param_names: list[str]) -> list[tuple[str, dict[str, float]]]:
    p = dict(zip(param_names, best_full))
    if model_name == "baseline_unfrozen":
        return [
            ("g2_zero", {"g2": 0.0}),
            ("powerCS_is_1", {"power_CS": 1.0}),
            ("a1_zero", {"a1": 0.0}),
            ("a2_zero", {"a2": 0.0}),
            ("a3_zero", {"a3": 0.0}),
            ("a4_zero", {"a4": 0.0}),
            ("b1_zero", {"b1": 0.0}),
            ("b2_zero", {"b2": 0.0}),
            ("b3_zero", {"b3": 0.0}),
            ("alpha_is_1", {"a": 1.0}),
        ]
    if model_name == "poly_bstar_cslog":
        return [
            ("lambda3_zero", {"lambda3": 0.0}),
            ("lambda4_zero", {"lambda4": 0.0}),
            ("alpha_is_1", {"alpha": 1.0}),
            ("c1_zero", {"c1": 0.0}),
        ]
    if model_name == "poly_bstar_cslog_loggauss":
        return [
            ("lambda3_zero", {"lambda3": 0.0}),
            ("lambda4_zero", {"lambda4": 0.0}),
            ("alpha_is_1", {"alpha": 1.0}),
            ("c1_zero", {"c1": 0.0}),
            ("bump_off", {"amp": 0.0, "logx0": p["logx0"], "sigx": p["sigx"]}),
        ]
    raise ValueError(model_name)


def parse_struct_fields(card_text: str) -> list[str]:
    match = re.search(r"struct\s+Params_Struct(.*?)end", card_text, re.S)
    if not match:
        raise RuntimeError("Could not find Params_Struct block")
    fields: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            fields.append(part.split("::", 1)[0].strip())
    return fields


def parse_last_bracket_block(text: str, name: str) -> str:
    pattern = rf"(?ms)^[ \t]*(?!#){re.escape(name)}\s*=\s*\[(.*?)\]"
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"Could not find {name}")
    return matches[-1]


def parse_array(text: str, name: str) -> list:
    raw = parse_last_bracket_block(text, name)
    cleaned = []
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            cleaned.append(stripped)
    joined = " ".join(cleaned)
    if not joined.strip():
        return []
    return list(ast.literal_eval("[" + joined + "]"))


def replace_bracket_assignment(text: str, name: str, values: list[float]) -> str:
    rendered = ", ".join(f"{v:.12g}" for v in values)
    pattern = rf"(?ms)^([ \t]*(?!#){re.escape(name)}\s*=\s*)\[(.*?)\]"
    return re.sub(pattern, rf"\1[{rendered}]", text, count=1)


def replace_scalar_assignment(text: str, name: str, value: str) -> str:
    pattern = rf'(?m)^([ \t]*(?!#){re.escape(name)}\s*=\s*)".*?"'
    return re.sub(pattern, rf'\1"{value}"', text, count=1)


def replace_mustar_func(kernel_text: str, n: int) -> str:
    pattern = r"(?ms)inline float mustar_func\(float b, float Q\)\s*\{.*?\n\}"
    return re.sub(pattern, mustar_function_src(n).rstrip(), kernel_text, count=1)


def load_best_full(result_path: Path) -> list[float]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def make_spec(model: ModelConfig) -> CandidateSpec:
    card_text = model.card_path.read_text(encoding="utf-8")
    kernel_text = model.kernel_path.read_text(encoding="utf-8")
    param_names = parse_struct_fields(card_text)
    best_full = load_best_full(model.result_path)
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]

    fit_name = f"Sig0p2_{model.name}"
    np_name = f"NP-{fit_name}.cl"

    card_text = replace_scalar_assignment(card_text, "const NP_name", np_name)
    card_text = replace_scalar_assignment(card_text, "const table_name", TARGET_TABLE)
    card_text = replace_bracket_assignment(card_text, "initial_params", best_full)
    kernel_text = replace_mustar_func(kernel_text, 0)

    (CARDS_DIR / f"{fit_name}.jl").write_text(card_text, encoding="utf-8")
    (NP_DIR / np_name).write_text(kernel_text, encoding="utf-8")

    return CandidateSpec(
        name=model.name,
        fit_name=fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=best_full,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="significance",
    )


def cleanup_generated_files() -> None:
    for model in MODELS:
        fit_name = f"Sig0p2_{model.name}"
        (CARDS_DIR / f"{fit_name}.jl").unlink(missing_ok=True)
        (NP_DIR / f"NP-{fit_name}.cl").unlink(missing_ok=True)


def finite_diff_hessian(
    objective_log,
    x: np.ndarray,
    bounds: tuple[float, float] = (0.0, 1.0),
    rel_step: float = 2e-4,
    abs_step: float = 1e-6,
):
    x = np.asarray(x, dtype=float)
    n = x.size
    lb = np.full(n, bounds[0], dtype=float)
    ub = np.full(n, bounds[1], dtype=float)

    def clamp(z: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(z, lb), ub)

    h = np.maximum(abs_step, rel_step * (np.abs(x) + 1.0))
    max_h = 0.49 * np.minimum(x - lb, ub - x)
    h = np.minimum(h, np.maximum(max_h, 0.0))

    cache: dict[bytes, float] = {}
    nevals = 0

    def feval(z: np.ndarray) -> float:
        nonlocal nevals
        z = clamp(np.asarray(z, dtype=float))
        key = z.tobytes()
        if key in cache:
            return cache[key]
        val = float(objective_log(z))
        cache[key] = val
        nevals += 1
        return val

    f0 = feval(x)
    dirs = np.zeros(n, dtype=int)
    for i in range(n):
        if h[i] == 0.0:
            dirs[i] = 0
        elif x[i] + h[i] <= ub[i] and x[i] - h[i] >= lb[i]:
            dirs[i] = 0
        elif x[i] + h[i] <= ub[i]:
            dirs[i] = 1
        elif x[i] - h[i] >= lb[i]:
            dirs[i] = -1
        else:
            dirs[i] = 0

    f_p = np.full(n, np.nan)
    f_m = np.full(n, np.nan)
    for i in range(n):
        if h[i] == 0:
            continue
        if dirs[i] == 0:
            xp = x.copy(); xp[i] += h[i]
            xm = x.copy(); xm[i] -= h[i]
            f_p[i] = feval(xp)
            f_m[i] = feval(xm)
        elif dirs[i] == 1:
            xp = x.copy(); xp[i] += h[i]
            f_p[i] = feval(xp)
        else:
            xm = x.copy(); xm[i] -= h[i]
            f_m[i] = feval(xm)

    g = np.zeros(n, dtype=float)
    H = np.zeros((n, n), dtype=float)

    for i in range(n):
        if h[i] == 0:
            continue
        if dirs[i] == 0:
            g[i] = (f_p[i] - f_m[i]) / (2.0 * h[i])
            H[i, i] = (f_p[i] - 2.0 * f0 + f_m[i]) / (h[i] ** 2)
        elif dirs[i] == 1:
            g[i] = (f_p[i] - f0) / h[i]
            x2p = x.copy(); x2p[i] += 2.0 * h[i]
            f2p = feval(x2p)
            H[i, i] = (f2p - 2.0 * f_p[i] + f0) / (h[i] ** 2)
        else:
            g[i] = (f0 - f_m[i]) / h[i]
            x2m = x.copy(); x2m[i] -= 2.0 * h[i]
            f2m = feval(x2m)
            H[i, i] = (f0 - 2.0 * f_m[i] + f2m) / (h[i] ** 2)

    for i in range(n):
        for j in range(i + 1, n):
            if h[i] == 0 or h[j] == 0:
                Hij = 0.0
            elif dirs[i] == 0 and dirs[j] == 0:
                xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                Hij = (feval(xpp) - feval(xpm) - feval(xmp) + feval(xmm)) / (4.0 * h[i] * h[j])
            else:
                si = 1.0 if dirs[i] >= 0 else -1.0
                sj = 1.0 if dirs[j] >= 0 else -1.0
                xij = x.copy(); xij[i] += si * h[i]; xij[j] += sj * h[j]
                xi = x.copy(); xi[i] += si * h[i]
                xj = x.copy(); xj[j] += sj * h[j]
                Hij = (feval(xij) - feval(xi) - feval(xj) + f0) / (si * sj * h[i] * h[j])
            H[i, j] = Hij
            H[j, i] = Hij

    return {"f0": f0, "grad": g, "H_log": H, "nevals": nevals}


def psd_covariance(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(0.5 * (H + H.T))
    floor = max(np.max(evals) * 1e-12, 1e-12)
    evals_psd = np.maximum(evals, floor)
    H_psd = (evecs * evals_psd) @ evecs.T
    cov = 2.0 * np.linalg.inv(H_psd)
    return H_psd, cov


def top_correlations(corr: np.ndarray, names: list[str], top_n: int = 10) -> list[dict[str, float | str]]:
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({"param_i": names[i], "param_j": names[j], "corr": float(corr[i, j]), "abs_corr": float(abs(corr[i, j]))})
    rows.sort(key=lambda row: row["abs_corr"], reverse=True)
    return rows[:top_n]


class NestedFit:
    def __init__(self, session: FitSession, best_full: np.ndarray, fixed_map: dict[str, float]) -> None:
        self.session = session
        self.best_full = np.asarray(best_full, dtype=float)
        self.fixed_map = fixed_map
        self.name_to_idx = {name: i for i, name in enumerate(session.spec.param_names)}
        self.fixed_idx = np.asarray(sorted({self.name_to_idx[name] for name in fixed_map} | set(session.frozen_idx.tolist())), dtype=int)
        self.mask = np.ones(len(self.best_full), dtype=bool)
        self.mask[self.fixed_idx] = False
        self.free_idx = np.where(self.mask)[0]
        self.bounds_free = session.bounds_full[self.free_idx]
        self.lower = self.bounds_free[:, 0]
        self.upper = self.bounds_free[:, 1]

        self.fixed_values = self.best_full.copy()
        for name, value in fixed_map.items():
            self.fixed_values[self.name_to_idx[name]] = float(value)

        self.theta0 = self.normalize(self.fixed_values[self.free_idx])
        self.rng = np.random.default_rng(20260322 + sum(ord(c) for c in ",".join(sorted(fixed_map))))

    def normalize(self, params_free: np.ndarray) -> np.ndarray:
        return (params_free - self.lower) / (self.upper - self.lower)

    def denormalize(self, theta: np.ndarray) -> np.ndarray:
        return self.lower + theta * (self.upper - self.lower)

    def full_from_free(self, params_free: np.ndarray) -> np.ndarray:
        full = self.fixed_values.copy()
        full[self.free_idx] = np.asarray(params_free, dtype=float)
        return full

    def evaluate_free(self, params_free: np.ndarray) -> dict:
        full = self.full_from_free(params_free)
        predictions, compute_s = self.session._predict(full)
        chi2dN, chi2_list, _ = self.session.get_chi2(predictions)
        return {
            "full_params": full.tolist(),
            "free_params": np.asarray(params_free, dtype=float).tolist(),
            "metrics": self.session.summary_metrics(predictions, chi2_list, compute_s),
            "per_file_chi2dN": {k: float(v) for k, v in chi2_list.items()},
        }

    def objective_log(self, theta: np.ndarray) -> float:
        params_free = self.denormalize(np.asarray(theta, dtype=float))
        value = max(float(self.evaluate_free(params_free)["metrics"]["chi2dN_total"]), 1e-12)
        return float(np.log10(value))

    def starts(self) -> list[np.ndarray]:
        starts = [self.theta0.copy()]
        for scale in (0.04, 0.08):
            starts.append(np.clip(self.theta0 + self.rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))
        uniq = []
        seen = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(start)
        return uniq

    def solve(self, maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1 = []
        stage1_budget = max(20, maxfun // 3)
        polish_budget = max(30, maxfun // 2)
        for start in self.starts():
            res = pybobyqa.solve(
                self.objective_log,
                start,
                bounds=(np.zeros_like(start), np.ones_like(start)),
                maxfun=stage1_budget,
                rhobeg=0.08,
                rhoend=3e-4,
                scaling_within_bounds=True,
                seek_global_minimum=False,
            )
            stage1.append({"theta": np.asarray(res.x, dtype=float), "log10": float(res.f), "nf": int(res.nf), "flag": int(res.flag)})
        stage1.sort(key=lambda item: item["log10"])
        best_start = stage1[0]
        polish = pybobyqa.solve(
            self.objective_log,
            best_start["theta"],
            bounds=(np.zeros_like(best_start["theta"]), np.ones_like(best_start["theta"])),
            maxfun=polish_budget,
            rhobeg=0.04,
            rhoend=1e-6,
            scaling_within_bounds=True,
            seek_global_minimum=False,
        )
        elapsed = time.perf_counter() - t0
        params_free = self.denormalize(np.asarray(polish.x, dtype=float))
        return {
            "free_params": params_free.tolist(),
            "nf": int(sum(item["nf"] for item in stage1) + int(polish.nf)),
            "elapsed_s": elapsed,
            "flag": int(polish.flag),
            "stage1_best_log10": float(best_start["log10"]),
            "polish_log10": float(polish.f),
        }


def analyze_model(model: ModelConfig, nested_maxfun: int) -> None:
    spec = make_spec(model)
    session = FitSession(spec)
    n_total = int(sum(session.n_list.values()))
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    best_full = np.asarray(best_eval["full_params"], dtype=float)
    best_free = np.asarray(best_eval["free_params"], dtype=float)

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0)
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10 ** 2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    H_psd, cov_norm = psd_covariance(H_total)

    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    sigma_norm = np.sqrt(np.maximum(np.diag(cov_norm), 0.0))
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    ref_map = REFERENCE_MAP[model.name]
    param_rows = []
    for name, value, sig_n, sig_p in zip(spec.param_names, best_full, sigma_norm, sigma_phys):
        row: dict[str, float | str | None] = {
            "model": model.name,
            "param": name,
            "value": float(value),
            "sigma_norm": float(sig_n),
            "sigma_phys": float(sig_p),
        }
        if name in ref_map and sig_p > 0:
            ref = float(ref_map[name])
            row["reference"] = ref
            row["z_vs_reference"] = float((value - ref) / sig_p)
        else:
            row["reference"] = np.nan
            row["z_vs_reference"] = np.nan
            if abs(value) > 1e-12:
                row["frac_uncertainty"] = float(sig_p / abs(value))
            else:
                row["frac_uncertainty"] = np.nan
        param_rows.append(row)

    corr_rows = top_correlations(corr, spec.param_names, top_n=12)

    nested_rows = []
    for test_name, fixed_map in null_tests_for(model.name, best_full.tolist(), spec.param_names):
        nested = NestedFit(session, best_full, fixed_map)
        fit_info = nested.solve(maxfun=nested_maxfun)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        metrics = best_nested["metrics"]
        nested_rows.append(
            {
                "model": model.name,
                "test": test_name,
                "fixed_map": json.dumps(fixed_map, sort_keys=True),
                "chi2dN_total": metrics["chi2dN_total"],
                "chi2_total": metrics["chi2dN_total"] * n_total,
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
                "delta_chi2dN": metrics["chi2dN_total"] - chi2dN,
                "delta_chi2_total": (metrics["chi2dN_total"] - chi2dN) * n_total,
                "delta_absdev": metrics["highE_mean_absdev_first3"] - best_eval["metrics"]["highE_mean_absdev_first3"],
                "delta_shortfall": metrics["highE_mean_shortfall_first3"] - best_eval["metrics"]["highE_mean_shortfall_first3"],
                "fit_evals": fit_info["nf"],
                "fit_elapsed_s": fit_info["elapsed_s"],
            }
        )

    base_summary = {
        "model": model.name,
        "label": model.label,
        "n_total": n_total,
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "highE_mean_absdev_first3": float(best_eval["metrics"]["highE_mean_absdev_first3"]),
        "highE_mean_shortfall_first3": float(best_eval["metrics"]["highE_mean_shortfall_first3"]),
        "hessian_evals": h["nevals"],
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
        "hessian_max_eig": float(np.max(np.linalg.eigvalsh(0.5 * (H_total + H_total.T)))),
    }

    (RESULTS_DIR / f"{model.name}_summary.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / f"{model.name}_parameter_uncertainty.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / f"{model.name}_top_correlations.csv", index=False)
    pd.DataFrame(nested_rows).sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).to_csv(
        RESULTS_DIR / f"{model.name}_nested_tests.csv", index=False
    )


def summarize() -> None:
    summary_rows = []
    nested_best_rows = []
    for model in MODELS:
        path = RESULTS_DIR / f"{model.name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary_rows.append(data)

        nested_path = RESULTS_DIR / f"{model.name}_nested_tests.csv"
        if nested_path.exists():
            df_nested = pd.read_csv(nested_path)
            strongest = df_nested.sort_values(["delta_chi2_total", "delta_absdev"], ascending=[False, True]).head(5).copy()
            strongest.insert(0, "rank", range(1, len(strongest) + 1))
            strongest.to_csv(RESULTS_DIR / f"{model.name}_nested_top5.csv", index=False)
            for _, row in strongest.iterrows():
                nested_best_rows.append(
                    {
                        "model": model.name,
                        "rank": int(row["rank"]),
                        "test": row["test"],
                        "delta_chi2_total": row["delta_chi2_total"],
                        "delta_absdev": row["delta_absdev"],
                        "delta_shortfall": row["delta_shortfall"],
                    }
                )

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("chi2dN_total").to_csv(RESULTS_DIR / "model_summaries.csv", index=False)
    if nested_best_rows:
        pd.DataFrame(nested_best_rows).to_csv(RESULTS_DIR / "nested_top5_all_models.csv", index=False)

    lines = [
        "# Parameter Significance On Default 0-2",
        "",
        "Method:",
        "",
        "- Local finite-difference Hessian around the fitted 0-2 point.",
        "- Nested local refits with one term turned off when there is a natural null/reference value.",
        "- `delta_chi2_total` is the main contribution score for the nested tests.",
        "",
        "Analyzed models:",
        "",
    ]
    for row in sorted(summary_rows, key=lambda item: item["chi2dN_total"]):
        lines.append(
            f"- `{row['model']}`: chi2/N `{row['chi2dN_total']:.6f}`, "
            f"highE absdev `{row['highE_mean_absdev_first3']:.6f}`, "
            f"highE shortfall `{row['highE_mean_shortfall_first3']:.6f}`"
        )
    lines.extend(["", "See per-model CSV files for parameter uncertainties, top correlations, and nested tests.", ""])
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def orchestrate(nested_maxfun: int) -> None:
    for model in MODELS:
        cmd = [
            str(Path(sys.executable)),
            str(Path(__file__)),
            "--model",
            model.name,
            "--nested-maxfun",
            str(nested_maxfun),
        ]
        print(f"\n=== Significance {model.name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"exit code {proc.returncode}")
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--nested-maxfun", type=int, default=60)
    args = parser.parse_args()

    cleanup_generated_files()
    try:
        if args.model:
            model = next(item for item in MODELS if item.name == args.model)
            analyze_model(model, nested_maxfun=args.nested_maxfun)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        else:
            orchestrate(nested_maxfun=args.nested_maxfun)
    finally:
        cleanup_generated_files()


if __name__ == "__main__":
    main()
