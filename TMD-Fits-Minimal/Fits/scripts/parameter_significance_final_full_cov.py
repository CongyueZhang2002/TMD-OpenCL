from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pybobyqa
from julia.api import Julia

from _paths import CARDS_DIR, RESULTS_ROOT, ROOT
from auto_np_search import EXPERIMENTS, FILE_EXCLUDES, JULIA_RUNTIME
from parameter_significance_0_2 import (
    finite_diff_hessian,
    parse_struct_fields,
    psd_covariance,
    top_correlations,
)


RESULTS_DIR = RESULTS_ROOT / "parameter_significance_final_full_cov_results"
RESULTS_DIR.mkdir(exist_ok=True)
CARD_PATH = CARDS_DIR / "Final.jl"

REFERENCE_MAP = {
    "lambda1": 0.0,
    "lambda2": 0.0,
    "lambda3": 0.0,
    "amp": 0.0,
    "c0": 0.0,
    "c1": 0.0,
}


def _norm_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def null_tests_for(best_full: list[float], param_names: list[str], frozen_idx: np.ndarray) -> list[tuple[str, dict[str, float]]]:
    name_to_idx = {name: i for i, name in enumerate(param_names)}
    frozen = set(int(i) for i in frozen_idx.tolist())
    p = dict(zip(param_names, best_full))
    tests: list[tuple[str, dict[str, float]]] = []

    for name in ("lambda1", "lambda2", "lambda3", "c0", "c1"):
        if name in name_to_idx and name_to_idx[name] not in frozen:
            tests.append((f"{name}_zero", {name: 0.0}))

    if "amp" in name_to_idx and name_to_idx["amp"] not in frozen:
        tests.append(("amp_zero", {"amp": 0.0}))

    if {"amp", "logx0", "sigx"}.issubset(name_to_idx) and name_to_idx["amp"] not in frozen:
        tests.append(("bump_off", {"amp": 0.0, "logx0": p["logx0"], "sigx": p["sigx"]}))

    deduped: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for name, fixed_map in tests:
        key = json.dumps([name, fixed_map], sort_keys=True)
        if key not in seen:
            deduped.append((name, fixed_map))
            seen.add(key)
    return deduped


class FullCovSession:
    def __init__(self, card_path: Path) -> None:
        self.jl = Julia(runtime=str(JULIA_RUNTIME), compiled_modules=False)
        from julia import Main

        self.Main = Main
        self.card_path = card_path
        self.card_text = card_path.read_text(encoding="utf-8")
        self.param_names = parse_struct_fields(self.card_text)

        self._include(str(card_path))
        self._include(str(ROOT / "DY" / f"DY_table_{self.Main.flavor_scheme}.jl"))

        self.approximate_total_xsec = True
        self.file_root = ROOT / "Data" / self.Main.data_name / "Cutted" / "DY"
        self.table_root_rel = Path("..") / "Tables" / self.Main.table_name / "DY"
        self.total_root = ROOT / "Data" / "DY_total_xsec" / f"{self.Main.pdf_name}.csv"
        self.chi2_root = ROOT / "Data" / "Chi2_Matrix"

        self.initial_params = np.asarray(self.Main.initial_params, dtype=float)
        self.frozen_idx = np.asarray(self.Main.frozen_indices, dtype=int)
        mask = np.ones(len(self.initial_params), dtype=bool)
        mask[self.frozen_idx] = False
        self.free_idx = np.where(mask)[0]
        self.frozen_vals = self.initial_params[self.frozen_idx].copy()
        self.free_param_names = [self.param_names[i] for i in self.free_idx]

        self.bounds_full = np.asarray(self.Main.bounds_raw, dtype=float)
        self.bounds_free = self.bounds_full[self.free_idx]
        self.lower_bounds = self.bounds_free[:, 0]
        self.upper_bounds = self.bounds_free[:, 1]
        self.theta0 = self.normalize_params(self.initial_params[self.free_idx])

        self.file_names = self._build_file_list()
        self.file_lengths = self._load_file_lengths()
        self.data_list = self._load_data()
        self.total_xsec_names = set(pd.read_csv(self.total_root)["name"].tolist())

        self.chi2_index_df = self._load_index()
        self.chi2_file_positions = {
            file_name: group["global_index"].to_numpy(dtype=int)
            for file_name, group in self.chi2_index_df.groupby("file", sort=False)
        }
        self.n_list = {file_name: len(positions) for file_name, positions in self.chi2_file_positions.items()}
        self.n_total = int(len(self.chi2_index_df))

        self.Total_inverse = pd.read_csv(self.chi2_root / "Total_inverse.csv").to_numpy(dtype=float)
        if self.Total_inverse.shape != (self.n_total, self.n_total):
            raise ValueError(
                f"Total_inverse.csv shape {self.Total_inverse.shape} does not match Index.csv length {self.n_total}"
            )
        self.Total_inverse = 0.5 * (self.Total_inverse + self.Total_inverse.T)
        self.data_vector = self._build_indexed_column_vector(
            {file_name: df["xsec"].to_numpy(dtype=float) for file_name, df in self.data_list.items()}
        )

        # Warm up the OpenCL path once.
        self._predict(self.initial_params)

    def _include(self, path: str) -> None:
        self.Main.eval(f'include(raw"{path}")')

    def _build_file_list(self) -> list[str]:
        file_names: list[str] = []
        for experiment in EXPERIMENTS:
            exp_dir = self.file_root / experiment
            for csv_path in sorted(exp_dir.glob("*.csv")):
                rel = Path(experiment, csv_path.name).as_posix()
                if rel in FILE_EXCLUDES:
                    continue
                file_names.append(rel)
        return file_names

    def _load_file_lengths(self) -> dict[str, int]:
        return {
            file_name: len(pd.read_csv(self.file_root / file_name))
            for file_name in self.file_names
        }

    def _load_data(self) -> dict[str, pd.DataFrame]:
        total_lookup = {
            row["name"]: float(row["total_xsec"])
            for _, row in pd.read_csv(self.total_root).iterrows()
        }
        data_list: dict[str, pd.DataFrame] = {}
        for file_name in self.file_names:
            df = pd.read_csv(self.file_root / file_name)
            stem = Path(file_name).stem
            if stem in total_lookup:
                df["total_xsec"] = total_lookup[stem] * np.ones(len(df))
            data_list[file_name] = df
        return data_list

    def _load_index(self) -> pd.DataFrame:
        index_df = pd.read_csv(self.chi2_root / "Index.csv")
        index_df["file"] = index_df["file"].astype(str).str.replace("\\", "/", regex=False)
        required_cols = {"global_index", "file", "local_index"}
        missing = required_cols - set(index_df.columns)
        if missing:
            raise ValueError(f"Index.csv is missing required columns: {sorted(missing)}")

        index_df = index_df.sort_values("global_index").reset_index(drop=True)
        expected_global = np.arange(len(index_df), dtype=int)
        actual_global = index_df["global_index"].to_numpy(dtype=int)
        if not np.array_equal(actual_global, expected_global):
            raise ValueError("Index.csv global_index must be contiguous and start at 0.")

        file_set = set(self.file_names)
        index_file_set = set(index_df["file"].unique())
        missing_from_index = sorted(file_set - index_file_set)
        extra_in_index = sorted(index_file_set - file_set)
        if missing_from_index:
            raise ValueError(f"Index.csv is missing active fit files: {missing_from_index}")
        if extra_in_index:
            raise ValueError(f"Index.csv contains files not used by this fit: {extra_in_index}")

        for file_name, group in index_df.groupby("file", sort=False):
            local_indices = group["local_index"].to_numpy(dtype=int)
            file_len = len(self.data_list[file_name])
            if np.any(local_indices < 0) or np.any(local_indices >= file_len):
                raise IndexError(f"Index.csv local_index out of bounds for {file_name}")

        return index_df

    def _build_indexed_column_vector(self, arrays_by_file: dict[str, np.ndarray]) -> np.ndarray:
        vector = np.full((len(self.chi2_index_df), 1), np.nan, dtype=float)
        for file_name, group in self.chi2_index_df.groupby("file", sort=False):
            values = np.asarray(arrays_by_file[file_name], dtype=float).reshape(-1)
            local_indices = group["local_index"].to_numpy(dtype=int)
            global_indices = group["global_index"].to_numpy(dtype=int)
            vector[global_indices, 0] = values[local_indices]
        if np.isnan(vector).any():
            raise ValueError("Indexed column vector has unfilled entries.")
        return vector

    def normalize_params(self, params_free: np.ndarray) -> np.ndarray:
        return (params_free - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def denormalize_params(self, theta: np.ndarray) -> np.ndarray:
        return self.lower_bounds + theta * (self.upper_bounds - self.lower_bounds)

    def full_from_free(self, params_free: np.ndarray) -> np.ndarray:
        full = self.initial_params.copy()
        full[self.free_idx] = np.asarray(params_free, dtype=float)
        full[self.frozen_idx] = self.frozen_vals
        return full

    def _prediction_reformat(self, predictions) -> dict[str, np.ndarray]:
        preds = {_norm_path(k): float(v) for k, v in predictions.items()}
        out: dict[str, np.ndarray] = {}

        for file_name in self.file_names:
            n_points = self.file_lengths[file_name]
            base = os.path.splitext(file_name)[0]
            xs = []
            for i in range(n_points):
                table_path = _norm_path(os.path.join(str(self.table_root_rel), f"{base}/{i}.jls"))
                xs.append(preds[table_path])
            arr = np.asarray(xs, dtype=float)

            if self.approximate_total_xsec and Path(file_name).stem in self.total_xsec_names:
                df = self.data_list[file_name]
                data_xsec = df["xsec"].to_numpy(dtype=float)
                qT_bin_size = df["qT_max"].to_numpy(dtype=float) - df["qT_min"].to_numpy(dtype=float)
                weighted_data = float(np.sum(data_xsec * qT_bin_size))
                weighted_theory = float(np.sum(arr * qT_bin_size))
                if weighted_theory != 0.0:
                    arr = (weighted_data / weighted_theory) * arr

            out[file_name] = arr

        return out

    def _predict(self, params_full: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        params_cl = self.Main.Params_Struct(*[np.float32(x) for x in params_full])
        self.Main.set_params(self.Main.VRAM, params_cl)
        predictions, compute_s = self.Main.xsec_dict(self.Main.rel_paths, self.Main.VRAM)
        return self._prediction_reformat(predictions), float(compute_s)

    def get_chi2(self, predictions: dict[str, np.ndarray]) -> tuple[float, dict[str, float], dict[str, int]]:
        prediction_vector = self._build_indexed_column_vector(predictions)
        diff_vector = self.data_vector - prediction_vector
        weighted_diff = self.Total_inverse @ diff_vector

        chi2_total = float((diff_vector.T @ weighted_diff)[0, 0])
        chi2dN_total = chi2_total / self.n_total

        point_contributions = diff_vector[:, 0] * weighted_diff[:, 0]
        chi2dN_by_file: dict[str, float] = {}
        for file_name in self.file_names:
            positions = self.chi2_file_positions[file_name]
            chi2_file = float(np.sum(point_contributions[positions]))
            chi2dN_by_file[file_name] = chi2_file / len(positions)

        return chi2dN_total, chi2dN_by_file, self.n_list

    def evaluate_free(self, params_free: np.ndarray) -> dict:
        full = self.full_from_free(params_free)
        predictions, compute_s = self._predict(full)
        chi2dN_total, chi2dN_by_file, _ = self.get_chi2(predictions)
        return {
            "full_params": full.tolist(),
            "free_params": np.asarray(params_free, dtype=float).tolist(),
            "metrics": {
                "chi2dN_total": float(chi2dN_total),
                "chi2_total": float(chi2dN_total * self.n_total),
                "compute_s": float(compute_s),
            },
            "per_file_chi2dN": {key: float(val) for key, val in chi2dN_by_file.items()},
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


class NestedFit:
    def __init__(self, session: FullCovSession, best_full: np.ndarray, fixed_map: dict[str, float]) -> None:
        self.session = session
        self.best_full = np.asarray(best_full, dtype=float)
        self.fixed_map = fixed_map
        self.name_to_idx = {name: i for i, name in enumerate(session.param_names)}
        self.fixed_idx = np.asarray(
            sorted({self.name_to_idx[name] for name in fixed_map} | set(session.frozen_idx.tolist())),
            dtype=int,
        )
        mask = np.ones(len(self.best_full), dtype=bool)
        mask[self.fixed_idx] = False
        self.free_idx = np.where(mask)[0]
        self.bounds_free = session.bounds_full[self.free_idx]
        self.lower = self.bounds_free[:, 0]
        self.upper = self.bounds_free[:, 1]

        self.fixed_values = self.best_full.copy()
        for name, value in fixed_map.items():
            self.fixed_values[self.name_to_idx[name]] = float(value)

        self.theta0 = self.normalize(self.fixed_values[self.free_idx])
        self.rng = np.random.default_rng(20260410 + sum(ord(c) for c in ",".join(sorted(fixed_map))))

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
        chi2dN_total, chi2dN_by_file, _ = self.session.get_chi2(predictions)
        return {
            "full_params": full.tolist(),
            "free_params": np.asarray(params_free, dtype=float).tolist(),
            "metrics": {
                "chi2dN_total": float(chi2dN_total),
                "chi2_total": float(chi2dN_total * self.session.n_total),
                "compute_s": float(compute_s),
            },
            "per_file_chi2dN": {key: float(val) for key, val in chi2dN_by_file.items()},
        }

    def objective_log(self, theta: np.ndarray) -> float:
        params_free = self.denormalize(np.asarray(theta, dtype=float))
        value = max(float(self.evaluate_free(params_free)["metrics"]["chi2dN_total"]), 1e-12)
        return float(np.log10(value))

    def starts(self) -> list[np.ndarray]:
        starts = [self.theta0.copy()]
        for scale in (0.04, 0.08):
            starts.append(np.clip(self.theta0 + self.rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))
        unique = []
        seen = set()
        for start in starts:
            key = tuple(np.round(start, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique

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
            stage1.append(
                {
                    "theta": np.asarray(res.x, dtype=float),
                    "log10": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )
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
            "elapsed_s": float(elapsed),
            "flag": int(polish.flag),
            "stage1_best_log10": float(best_start["log10"]),
            "polish_log10": float(polish.f),
        }


def max_abs_corr(corr: np.ndarray) -> float:
    if corr.size == 0:
        return float("nan")
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    return float(np.max(vals)) if vals.size else float("nan")


def write_summary_md(results_dir: Path, summary: dict, param_rows: list[dict], corr_rows: list[dict], nested_df: pd.DataFrame) -> None:
    lines = [
        "# Final Full-Covariance Significance",
        "",
        f"- card: `{CARD_PATH.name}`",
        f"- chi2/N: `{summary['chi2dN_total']:.6f}`",
        f"- chi2 total: `{summary['chi2_total']:.6f}`",
        f"- hessian evals: `{summary['hessian_evals']}`",
        f"- grad norm (log10 objective): `{summary['grad_norm_log10']:.6g}`",
        f"- hessian min eig: `{summary['hessian_min_eig']:.6g}`",
        f"- hessian max eig: `{summary['hessian_max_eig']:.6g}`",
        f"- max |corr|: `{summary['max_abs_corr']:.6f}`",
        "",
        "Top correlations:",
        "",
    ]
    for row in corr_rows[:8]:
        lines.append(f"- `{row['param_i']}` vs `{row['param_j']}`: `{row['corr']:.6f}`")
    lines.extend(["", "Most constrained parameters by |z vs reference|:", ""])

    ranked = [
        row for row in param_rows
        if np.isfinite(row.get("signal_to_sigma", np.nan))
    ]
    ranked.sort(key=lambda row: float(row["signal_to_sigma"]), reverse=True)
    for row in ranked[:8]:
        lines.append(
            f"- `{row['param']}`: value `{row['value']:.6g}`, sigma `{row['sigma_phys']:.6g}`, "
            f"z `{row['z_vs_reference']:.3f}`"
        )

    if not nested_df.empty:
        lines.extend(["", "Top nested tests by Δchi2:", ""])
        for _, row in nested_df.head(8).iterrows():
            lines.append(
                f"- `{row['test']}`: Δchi2 `{row['delta_chi2_total']:.6f}`, "
                f"nested chi2/N `{row['chi2dN_total']:.6f}`"
            )

    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_report(results_dir: Path, summary: dict, param_rows: list[dict], corr_rows: list[dict], nested_df: pd.DataFrame) -> None:
    lines = [
        "# Final Full-Covariance Significance",
        "",
        f"- card: `{CARD_PATH.name}`",
        f"- card chi2/N: `{summary['card_chi2dN_total']:.6f}`",
        f"- fitted chi2/N: `{summary['chi2dN_total']:.6f}`",
        f"- delta chi2 total from card to fitted point: `{summary['delta_chi2_total_from_card']:.6f}`",
        f"- chi2 total: `{summary['chi2_total']:.6f}`",
        f"- local refit evals: `{summary['refit_evals']}`",
        f"- local refit flag: `{summary['refit_flag']}`",
        f"- hessian evals: `{summary['hessian_evals']}`",
        f"- grad norm (log10 objective): `{summary['grad_norm_log10']:.6g}`",
        f"- hessian min eig: `{summary['hessian_min_eig']:.6g}`",
        f"- hessian max eig: `{summary['hessian_max_eig']:.6g}`",
        f"- max |corr|: `{summary['max_abs_corr']:.6f}`",
        "",
        "Top correlations:",
        "",
    ]
    for row in corr_rows[:8]:
        lines.append(f"- `{row['param_i']}` vs `{row['param_j']}`: `{row['corr']:.6f}`")
    lines.extend(["", "Most constrained parameters by |z vs reference|:", ""])

    ranked = [row for row in param_rows if np.isfinite(row.get("signal_to_sigma", np.nan))]
    ranked.sort(key=lambda row: float(row["signal_to_sigma"]), reverse=True)
    for row in ranked[:8]:
        lines.append(
            f"- `{row['param']}`: value `{row['value']:.6g}`, sigma `{row['sigma_phys']:.6g}`, "
            f"z `{row['z_vs_reference']:.3f}`"
        )

    if not nested_df.empty:
        lines.extend(["", "Top nested tests by Delta chi2:", ""])
        for _, row in nested_df.head(8).iterrows():
            lines.append(
                f"- `{row['test']}`: Delta chi2 `{row['delta_chi2_total']:.6f}`, "
                f"nested chi2/N `{row['chi2dN_total']:.6f}`"
            )

    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-maxfun", type=int, default=240)
    parser.add_argument("--nested-maxfun", type=int, default=120)
    parser.add_argument("--rel-step", type=float, default=2e-4)
    args = parser.parse_args()

    session = FullCovSession(CARD_PATH)
    card_eval = session.evaluate_free(session.initial_params[session.free_idx])
    card_full = np.asarray(card_eval["full_params"], dtype=float)
    best_eval = card_eval
    best_full = card_full
    refit_info = {
        "free_params": card_full[session.free_idx].tolist(),
        "nf": 0,
        "elapsed_s": 0.0,
        "flag": 0,
        "stage1_best_log10": float(np.log10(max(card_eval["metrics"]["chi2dN_total"], 1e-12))),
        "polish_log10": float(np.log10(max(card_eval["metrics"]["chi2dN_total"], 1e-12))),
    }

    print(f"[eval] card chi2/N = {card_eval['metrics']['chi2dN_total']:.6f}")
    if len(session.free_idx) > 0:
        print("[refit] local full-parameter polish", flush=True)
        refit = NestedFit(session, card_full, {})
        refit_info = refit.solve(maxfun=args.refit_maxfun)
        best_eval = refit.evaluate_free(np.asarray(refit_info["free_params"], dtype=float))
        best_full = np.asarray(best_eval["full_params"], dtype=float)
        print(f"[refit] fitted chi2/N = {best_eval['metrics']['chi2dN_total']:.6f}", flush=True)

    print("[hessian] finite differences", flush=True)
    theta_best = session.normalize_params(best_full[session.free_idx])
    h = finite_diff_hessian(session.objective_log_normalized, theta_best, rel_step=args.rel_step)

    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = float(best_eval["metrics"]["chi2_total"])
    n_total = session.n_total
    ln10 = math.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10**2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    H_psd, cov_norm = psd_covariance(H_total)
    diag = np.maximum(np.diag(cov_norm), 1e-300)
    corr = cov_norm / np.sqrt(np.outer(diag, diag))
    corr = np.clip(corr, -1.0, 1.0)

    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    sigma_norm = np.sqrt(np.maximum(np.diag(cov_norm), 0.0))

    free_name_to_idx = {name: i for i, name in enumerate(session.free_param_names)}
    param_rows = []
    for name, value in zip(session.param_names, best_full):
        ref = REFERENCE_MAP.get(name, np.nan)
        row: dict[str, float | str | bool] = {
            "param": name,
            "value": float(value),
            "reference": float(ref) if not pd.isna(ref) else np.nan,
            "is_free": name in free_name_to_idx,
        }
        if name in free_name_to_idx:
            idx = free_name_to_idx[name]
            sig_n = float(sigma_norm[idx])
            sig_p = float(sigma_phys[idx])
            row["sigma_norm"] = sig_n
            row["sigma_phys"] = sig_p
            row["frac_uncertainty"] = float(sig_p / abs(value)) if abs(value) > 1e-12 else np.nan
            if not pd.isna(ref) and sig_p > 0:
                z = (float(value) - float(ref)) / sig_p
                row["z_vs_reference"] = float(z)
                row["signal_to_sigma"] = float(abs(z))
            else:
                row["z_vs_reference"] = np.nan
                row["signal_to_sigma"] = np.nan
        else:
            row["sigma_norm"] = np.nan
            row["sigma_phys"] = np.nan
            row["frac_uncertainty"] = np.nan
            row["z_vs_reference"] = np.nan
            row["signal_to_sigma"] = np.nan
        param_rows.append(row)

    corr_rows = top_correlations(corr, session.free_param_names, top_n=16)

    nested_rows = []
    for test_name, fixed_map in null_tests_for(best_full.tolist(), session.param_names, session.frozen_idx):
        print(f"[nested] {test_name}", flush=True)
        nested = NestedFit(session, best_full, fixed_map)
        fit_info = nested.solve(maxfun=args.nested_maxfun)
        best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
        metrics = best_nested["metrics"]
        nested_rows.append(
            {
                "test": test_name,
                "fixed_map": json.dumps(fixed_map, sort_keys=True),
                "chi2dN_total": float(metrics["chi2dN_total"]),
                "chi2_total": float(metrics["chi2_total"]),
                "delta_chi2dN": float(metrics["chi2dN_total"] - chi2dN),
                "delta_chi2_total": float(metrics["chi2_total"] - chi2_total),
                "fit_evals": int(fit_info["nf"]),
                "fit_elapsed_s": float(fit_info["elapsed_s"]),
                "flag": int(fit_info["flag"]),
            }
        )

    eigvals = np.linalg.eigvalsh(0.5 * (H_total + H_total.T))
    top_corr_pair = corr_rows[0] if corr_rows else {"param_i": "", "param_j": "", "corr": np.nan}
    summary = {
        "card": str(CARD_PATH),
        "card_chi2dN_total": float(card_eval["metrics"]["chi2dN_total"]),
        "card_chi2_total": float(card_eval["metrics"]["chi2_total"]),
        "chi2dN_total": chi2dN,
        "chi2_total": chi2_total,
        "delta_chi2dN_from_card": float(best_eval["metrics"]["chi2dN_total"] - card_eval["metrics"]["chi2dN_total"]),
        "delta_chi2_total_from_card": float(best_eval["metrics"]["chi2_total"] - card_eval["metrics"]["chi2_total"]),
        "n_total": int(n_total),
        "refit_evals": int(refit_info["nf"]),
        "refit_elapsed_s": float(refit_info["elapsed_s"]),
        "refit_flag": int(refit_info["flag"]),
        "hessian_evals": int(h["nevals"]),
        "grad_norm_log10": float(np.linalg.norm(h["grad"])),
        "hessian_min_eig": float(np.min(eigvals)),
        "hessian_max_eig": float(np.max(eigvals)),
        "max_abs_corr": max_abs_corr(corr),
        "top_corr_param_i": top_corr_pair.get("param_i", ""),
        "top_corr_param_j": top_corr_pair.get("param_j", ""),
        "top_corr": float(top_corr_pair.get("corr", np.nan)),
        "rel_step": float(args.rel_step),
        "refit_maxfun": int(args.refit_maxfun),
        "nested_maxfun": int(args.nested_maxfun),
    }

    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / "parameter_uncertainty.csv", index=False)
    pd.DataFrame(cov_norm, index=session.free_param_names, columns=session.free_param_names).to_csv(
        RESULTS_DIR / "covariance_normalized.csv"
    )
    pd.DataFrame(cov_phys, index=session.free_param_names, columns=session.free_param_names).to_csv(
        RESULTS_DIR / "covariance_physical.csv"
    )
    pd.DataFrame(corr, index=session.free_param_names, columns=session.free_param_names).to_csv(
        RESULTS_DIR / "correlation_matrix.csv"
    )
    pd.DataFrame(H_total, index=session.free_param_names, columns=session.free_param_names).to_csv(
        RESULTS_DIR / "hessian_total.csv"
    )
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "top_correlations.csv", index=False)
    nested_df = pd.DataFrame(nested_rows).sort_values("delta_chi2_total", ascending=False)
    nested_df.to_csv(RESULTS_DIR / "nested_tests.csv", index=False)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "best_fit.json").write_text(
        json.dumps(
            {
                "card_eval": card_eval,
                "best_eval": best_eval,
                "refit": refit_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_report(RESULTS_DIR, summary, param_rows, corr_rows, nested_df)

    print(f"[done] wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
