from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_ROOT = REPO_ROOT / "Fits"
NOTEBOOK_PATH = FITS_ROOT / "fit_replicas_cov_full.ipynb"
CHI2_ROOT = REPO_ROOT / "Data" / "Chi2_Matrix"


def _load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _patched_source(src: str, temp_name: str) -> str:
    src = src.replace('output_csv_name = "replica_0410.csv"', f'output_csv_name = r"{temp_name}"')
    src = src.replace("use_random_seed = True", "use_random_seed = False")
    return src


def load_setup_namespace() -> dict:
    nb = _load_notebook()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_name = Path(tmp.name).name

    ns = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
    old_cwd = Path.cwd()
    try:
        os.chdir(FITS_ROOT)
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "replica_results = []" in src:
                break
            src = _patched_source(src, temp_name)
            exec(compile(src, f"{NOTEBOOK_PATH.name}:cell{idx}", "exec"), ns)
    finally:
        os.chdir(old_cwd)

    return ns


def build_central_predictions(ns: dict):
    params = ns["initial_params"]
    params_cl = ns["Main"].Params_Struct(*[np.float32(x) for x in params])
    ns["Main"].set_params(ns["Main"].VRAM, params_cl)
    predictions, _ = ns["Main"].xsec_dict(ns["Main"].rel_paths, ns["Main"].VRAM)
    return ns["prediction_reformat"](predictions)


def check_pdf_rep_factorization(ns: dict) -> None:
    pdf_cov = ns["to_float64"](pd.read_csv(CHI2_ROOT / "PDF_cov.csv")).to_numpy(dtype=float)
    recon = ns["PDF_rep"] @ ns["PDF_rep"].T
    max_abs = float(np.max(np.abs(recon - pdf_cov)))
    rel_fro = float(np.linalg.norm(recon - pdf_cov) / np.linalg.norm(pdf_cov))
    if max_abs > 1e-10 or rel_fro > 1e-10:
        raise AssertionError(f"PDF_rep factorization mismatch: max_abs={max_abs}, rel_fro={rel_fro}")
    print(f"PASS pdf_rep_factorization max_abs={max_abs:.3e} rel_fro={rel_fro:.3e}")


def check_pdf_seed_reproducibility(ns: dict) -> None:
    label, shift, seed = ns["sample_pdf_replica"](np.random.default_rng(123), 0)
    rebuilt_z = np.random.default_rng(seed).standard_normal(ns["PDF_rep"].shape[1])
    rebuilt_shift = (ns["PDF_rep"] @ rebuilt_z).reshape(len(ns["chi2_index_df"]), 1)
    max_abs = float(np.max(np.abs(rebuilt_shift - shift)))
    if label != "gaussian" or max_abs > 1e-12:
        raise AssertionError(f"PDF shift reproducibility failed: label={label}, max_abs={max_abs}")
    print(f"PASS pdf_seed_reproducibility seed={seed} max_abs={max_abs:.3e}")


def check_pdf_shift_enters_chi2(ns: dict, df_predictions) -> None:
    zero_shift = ns["zero_pdf_shift_vector"].copy()
    _, sampled_shift, _ = ns["sample_pdf_replica"](np.random.default_rng(456), 0)
    chi2_zero, _, _ = ns["get_chi2dN"](df_predictions, zero_shift, ns["nominal_data_list"])
    chi2_shift, _, _ = ns["get_chi2dN"](df_predictions, sampled_shift, ns["nominal_data_list"])
    delta = float(chi2_shift - chi2_zero)
    if abs(delta) < 1e-3:
        raise AssertionError(f"PDF shift did not change chi2 enough: delta={delta}")
    original_data_list = ns["data_list"]
    original_shift = ns["current_pdf_shift_vector"]
    try:
        ns["data_list"] = ns["nominal_data_list"]
        ns["current_pdf_shift_vector"] = zero_shift
        objective_zero = float(ns["objective"](ns["initial_params"]))
        ns["current_pdf_shift_vector"] = sampled_shift
        objective_shift = float(ns["objective"](ns["initial_params"]))
    finally:
        ns["data_list"] = original_data_list
        ns["current_pdf_shift_vector"] = original_shift
    if abs(objective_zero - chi2_zero) > 1e-5 or abs(objective_shift - chi2_shift) > 1e-5:
        raise AssertionError(
            "Objective path does not match direct chi2 calculation for PDF shift test: "
            f"objective_zero={objective_zero}, chi2_zero={chi2_zero}, "
            f"objective_shift={objective_shift}, chi2_shift={chi2_shift}"
        )
    print(f"PASS pdf_shift_enters_chi2 chi2_zero={chi2_zero:.6f} chi2_shift={chi2_shift:.6f} delta={delta:.6f}")


def check_replica_data_enters_chi2(ns: dict, df_predictions) -> None:
    zero_shift = ns["zero_pdf_shift_vector"].copy()
    nominal_chi2, _, _ = ns["get_chi2dN"](df_predictions, zero_shift, ns["nominal_data_list"])

    rng = np.random.default_rng(20260410)
    replica_data = ns["generate_experimental_replica_data"](rng)
    replica_chi2, _, _ = ns["get_chi2dN"](df_predictions, zero_shift, replica_data)
    delta = float(replica_chi2 - nominal_chi2)

    max_data_shift = 0.0
    for file in ns["file_names"]:
        nominal = ns["nominal_data_list"][file]["xsec"].to_numpy(dtype=float)
        current = replica_data[file]["xsec"].to_numpy(dtype=float)
        max_data_shift = max(max_data_shift, float(np.max(np.abs(current - nominal))))

    if max_data_shift <= 0.0 or abs(delta) < 1e-2:
        raise AssertionError(
            f"Replica data did not affect chi2 enough: max_data_shift={max_data_shift}, delta={delta}"
        )

    original_data_list = ns["data_list"]
    original_shift = ns["current_pdf_shift_vector"]
    try:
        ns["data_list"] = replica_data
        ns["current_pdf_shift_vector"] = zero_shift
        replica_df_predictions = build_central_predictions(ns)
        expected_replica_chi2, _, _ = ns["get_chi2dN"](replica_df_predictions, zero_shift, replica_data)
        objective_chi2 = float(ns["objective"](ns["initial_params"]))
    finally:
        ns["data_list"] = original_data_list
        ns["current_pdf_shift_vector"] = original_shift

    mismatch = abs(objective_chi2 - expected_replica_chi2)
    if mismatch > 1e-5:
        raise AssertionError(
            f"Objective is not using current replica data: objective={objective_chi2}, expected={expected_replica_chi2}"
        )

    print(
        "PASS replica_data_enters_chi2 "
        f"nominal={nominal_chi2:.6f} replica_fixed_pred={replica_chi2:.6f} "
        f"replica_objective={objective_chi2:.6f} delta={delta:.6f} max_data_shift={max_data_shift:.6f}"
    )


def check_central_replica_chi2_scale(ns: dict, df_predictions) -> None:
    rng = np.random.default_rng(12345)
    vals = []
    original_data_list = ns["data_list"]
    original_shift = ns["current_pdf_shift_vector"]
    try:
        for rid in range(10):
            replica_data = ns["generate_experimental_replica_data"](rng)
            _, pdf_shift, _ = ns["sample_pdf_replica"](rng, rid)
            ns["data_list"] = replica_data
            ns["current_pdf_shift_vector"] = pdf_shift
            vals.append(float(ns["objective"](ns["initial_params"])))
    finally:
        ns["data_list"] = original_data_list
        ns["current_pdf_shift_vector"] = original_shift

    vals = np.asarray(vals, dtype=float)
    mean_val = float(np.mean(vals))
    if mean_val < 1.5:
        raise AssertionError(f"Replica central-parameter chi2 mean looks too low: {mean_val}")

    print(
        "PASS central_replica_chi2_scale "
        f"mean={mean_val:.6f} min={float(np.min(vals)):.6f} max={float(np.max(vals)):.6f}"
    )


def check_fixed_seed_replica_rng(ns: dict) -> None:
    original_use_random_seed = ns["use_random_seed"]
    original_seed = ns["replica_seed"]
    try:
        ns["use_random_seed"] = False
        ns["replica_seed"] = 12345

        sequential_payloads = []
        for rid in range(6):
            rng = ns["make_replica_rng"](rid)
            replica_data = ns["generate_experimental_replica_data"](rng)
            _, pdf_shift, pdf_seed = ns["sample_pdf_replica"](rng, rid)
            starts = ns["build_replica_starts"](rng)
            sequential_payloads.append((replica_data, pdf_shift.copy(), pdf_seed, [s.copy() for s in starts]))

        direct_rng = ns["make_replica_rng"](5)
        direct_data = ns["generate_experimental_replica_data"](direct_rng)
        _, direct_shift, direct_seed = ns["sample_pdf_replica"](direct_rng, 5)
        direct_starts = ns["build_replica_starts"](direct_rng)

        seq_data, seq_shift, seq_seed, seq_starts = sequential_payloads[5]

        if seq_seed != direct_seed:
            raise AssertionError(f"Fixed-seed PDF seed mismatch on resume-safe generation: {seq_seed} vs {direct_seed}")

        max_shift_diff = float(np.max(np.abs(seq_shift - direct_shift)))
        if max_shift_diff > 1e-12:
            raise AssertionError(f"Fixed-seed PDF shift mismatch on resume-safe generation: {max_shift_diff}")

        max_data_diff = 0.0
        for file in ns["file_names"]:
            seq_vals = seq_data[file]["xsec"].to_numpy(dtype=float)
            direct_vals = direct_data[file]["xsec"].to_numpy(dtype=float)
            max_data_diff = max(max_data_diff, float(np.max(np.abs(seq_vals - direct_vals))))
        if max_data_diff > 1e-12:
            raise AssertionError(f"Fixed-seed data replica mismatch on resume-safe generation: {max_data_diff}")

        max_start_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(seq_starts, direct_starts))
        if max_start_diff > 1e-12:
            raise AssertionError(f"Fixed-seed start jitter mismatch on resume-safe generation: {max_start_diff}")
    finally:
        ns["use_random_seed"] = original_use_random_seed
        ns["replica_seed"] = original_seed

    print("PASS fixed_seed_replica_rng replica_id-specific RNG is stable")


def main() -> None:
    ns = load_setup_namespace()
    df_predictions = build_central_predictions(ns)

    check_pdf_rep_factorization(ns)
    check_pdf_seed_reproducibility(ns)
    check_pdf_shift_enters_chi2(ns, df_predictions)
    check_replica_data_enters_chi2(ns, df_predictions)
    check_central_replica_chi2_scale(ns, df_predictions)
    check_fixed_seed_replica_rng(ns)

    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
