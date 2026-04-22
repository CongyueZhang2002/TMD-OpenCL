from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from julia import Main
from tqdm.auto import tqdm


TMDPDF_TARGET_NAMES = ["f_u", "f_ub", "f_d", "f_db", "f_s", "f_sb", "f_c", "f_cb", "f_b", "f_bb"]
JULIA_TMDPDF_TARGET_NAMES = "[" + ", ".join(f'"{name}"' for name in TMDPDF_TARGET_NAMES) + "]"
FLAVORS = ["u", "ub", "d", "db", "s", "sb", "c", "cb", "b", "bb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one TMD band chunk in a fresh Python/Julia process.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file written by TMD_band_generator.ipynb")
    parser.add_argument("--chunk-csv", required=True, help="Path to the chunk CSV with assigned effective_pdf_replica_id")
    parser.add_argument("--output", required=True, help="Path to the chunk pickle output")
    return parser.parse_args()


def include(repo_root: Path, rel_path: str) -> None:
    path = (repo_root / rel_path).resolve()
    Main.eval(f'include(raw"{path}")')


def push_params(param_names: list[str], params: np.ndarray) -> None:
    lines = [f"global NP_{name} = Float32({float(value)})" for name, value in zip(param_names, params)]
    Main.eval("\n".join(lines))


def as_array(julia_tuple) -> np.ndarray:
    return np.asarray(julia_tuple, dtype=float)


def load_card_metadata(card_path: Path) -> tuple[list[str], np.ndarray]:
    card_text = card_path.read_text(encoding="utf-8")
    struct_match = re.search(r"struct\s+Params_Struct(.*?)end", card_text, re.S)
    if struct_match is None:
        raise ValueError(f"Could not find Params_Struct in {card_path}")

    param_names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*::\s*Float32", struct_match.group(1))
    init_matches = re.findall(r"(?ms)^\s*initial_params\s*=\s*\[([^\]]*)\]", card_text)
    if not init_matches:
        raise ValueError(f"Could not find initial_params in {card_path}")
    initial_params = np.asarray(
        [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", init_matches[-1])],
        dtype=float,
    )
    return param_names, initial_params


def load_chunk_df(chunk_csv: Path, param_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(chunk_csv)
    if df.empty:
        raise ValueError(f"{chunk_csv} does not contain any rows")

    working = df.copy()
    param_columns = [f"param_{i}" for i in range(len(param_names))]

    if all(col in working.columns for col in param_columns):
        for col in param_columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
        for col, name in zip(param_columns, param_names):
            working[name] = working[col]
    elif all(name in working.columns for name in param_names):
        for name in param_names:
            working[name] = pd.to_numeric(working[name], errors="coerce")
    else:
        raise ValueError(
            f"{chunk_csv} must contain either {param_columns} or the named parameter columns {param_names}"
        )

    required_columns = ["replica_id", "effective_pdf_replica_id", *param_names]
    missing_required = [col for col in required_columns if col not in working.columns]
    if missing_required:
        raise ValueError(f"{chunk_csv} is missing required columns: {missing_required}")

    working["replica_id"] = pd.to_numeric(working["replica_id"], errors="coerce")
    working["effective_pdf_replica_id"] = pd.to_numeric(working["effective_pdf_replica_id"], errors="coerce")
    for col in ["input_pdf_replica_id", "source_index", "success", "nfev"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    if "best_chi2dN" in working.columns:
        working["best_chi2dN"] = pd.to_numeric(working["best_chi2dN"], errors="coerce")
    if "log_prob" in working.columns:
        working["log_prob"] = pd.to_numeric(working["log_prob"], errors="coerce")

    working = working.dropna(subset=["replica_id", "effective_pdf_replica_id", *param_names]).copy()
    working["replica_id"] = working["replica_id"].astype(int)
    working["effective_pdf_replica_id"] = working["effective_pdf_replica_id"].astype(int)
    return working.sort_values("replica_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    chunk_csv = Path(args.chunk_csv).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(config["repo_root"]).resolve()
    card_path = Path(config["card_path"]).resolve()
    tmd_root = (repo_root / "TMDs").resolve()

    param_names, initial_params = load_card_metadata(card_path)
    chunk_df = load_chunk_df(chunk_csv, param_names)
    if chunk_df.empty:
        raise ValueError(f"{chunk_csv} contains no valid rows after filtering")

    include(repo_root, f"Cards/{config['fit_name']}.jl")
    push_params(param_names, initial_params)

    np_cl_name = str(Main.NP_name)
    np_julia_name = Path(np_cl_name).with_suffix(".jl").name
    np_julia_path = tmd_root / "NP Parameterizations Julia" / np_julia_name
    if not np_julia_path.exists():
        raise FileNotFoundError(f"No Julia NP file found for {np_cl_name}. Expected {np_julia_path.name}.")

    Main.if_grid = True
    try:
        Main.FastGK
    except Exception:
        Main.eval("module FastGK end")
    try:
        Main.eval("b0")
    except Exception:
        Main.eval("const b0 = 1.1229189")

    include(repo_root, "TMDs/Grids/initialization.jl")
    include(repo_root, f"TMDs/NP Parameterizations Julia/{np_julia_name}")
    push_params(param_names, initial_params)
    include(repo_root, "TMDs/TMDPDFs/TMDPDFN.jl")
    include(repo_root, "Replica Analysis/TMD_band_generator.jl")

    table_name = str(Main.table_name)
    tmdpdf_plot_dir = tmd_root / "Grids" / table_name / "TMDPDF_plot"
    if not tmdpdf_plot_dir.exists():
        raise FileNotFoundError(f"Missing TMDPDF_plot directory: {tmdpdf_plot_dir}")

    available_pdf_grid_paths = {
        int(path.stem): path.resolve()
        for path in sorted(tmdpdf_plot_dir.glob("*.csv"), key=lambda p: int(p.stem))
        if path.stem.isdigit()
    }
    if not available_pdf_grid_paths:
        raise FileNotFoundError(f"No numeric TMDPDF_plot/*.csv grids found under {tmdpdf_plot_dir}")

    current_pdf_grid_path: Path | None = None

    def set_tmdpdf_grid_from_member(pdf_replica_id: int) -> None:
        nonlocal current_pdf_grid_path
        if pdf_replica_id not in available_pdf_grid_paths:
            raise KeyError(
                f"No TMDPDF_plot grid found for pdf_replica_id={pdf_replica_id} under {tmdpdf_plot_dir}"
            )
        grid_path = available_pdf_grid_paths[pdf_replica_id]
        if current_pdf_grid_path == grid_path:
            return
        Main.eval(
            f"""
            begin
                local df = DataFrame(CSV.File(raw"{grid_path}"))
                global df_TMDPDF = df
                global TMDPDF_bmin = Float64(minimum(df[!, "b"]))
                global TMDPDF_bmax = Float64(maximum(df[!, "b"]))
                initialize_interpolator(
                    df = df,
                    interpolator_name = "xTMDPDF_raw_grid",
                    variable_names = ["x", "b"],
                    target_names = {JULIA_TMDPDF_TARGET_NAMES},
                )
                let itp = interpolators[:xTMDPDF_raw_grid]
                    global xTMDPDF_raw_grid
                    @inline xTMDPDF_raw_grid(x::Real, b::Real) = itp(x, b)
                end
            end
            """
        )
        current_pdf_grid_path = grid_path

    flavor_idx = FLAVORS.index(str(config["map_plot_flavor"]))
    kt_plot = np.linspace(float(config["kt_plot_min"]), float(config["kt_plot_max"]), int(config["n_kt_plot"]))
    map_x_values = [float(x) for x in config["map_x_values"]]
    map_q_values = [float(q) for q in config["map_Q_values"]]
    curve_keys = [(float(q), float(x)) for q in map_q_values for x in map_x_values]

    def evaluate_curve_bundle(params: np.ndarray) -> dict[tuple[float, float], np.ndarray]:
        push_params(param_names, params)
        kt_curves: dict[tuple[float, float], np.ndarray] = {}
        for q_value in map_q_values:
            for x_value in map_x_values:
                key = (float(q_value), float(x_value))
                values = as_array(
                    Main.TMDPDF_kt_vec(
                        kt_vec=np.asarray(kt_plot, dtype=np.float64),
                        x=float(x_value),
                        Q=float(q_value),
                    )
                )
                kt_curves[key] = np.asarray(x_value * values[flavor_idx, :], dtype=float)
        return kt_curves

    kt_samples = {key: [] for key in curve_keys}
    assignment_rows = []

    start_time = time.perf_counter()
    for row in tqdm(chunk_df.itertuples(index=False), total=len(chunk_df), desc=f"Chunk {chunk_csv.stem}"):
        params = np.array([getattr(row, name) for name in param_names], dtype=float)
        effective_pdf_replica_id = int(row.effective_pdf_replica_id)
        set_tmdpdf_grid_from_member(effective_pdf_replica_id)
        kt_curves = evaluate_curve_bundle(params)

        for key in curve_keys:
            kt_samples[key].append(kt_curves[key])

        assignment_rows.append(
            {
                "replica_id": int(row.replica_id),
                "source_index": None if not hasattr(row, "source_index") or pd.isna(row.source_index) else int(row.source_index),
                "log_prob": None if not hasattr(row, "log_prob") or pd.isna(row.log_prob) else float(row.log_prob),
                "input_pdf_replica_id": None
                if not hasattr(row, "input_pdf_replica_id") or pd.isna(row.input_pdf_replica_id)
                else int(row.input_pdf_replica_id),
                "effective_pdf_replica_id": effective_pdf_replica_id,
                "pdf_assignment_source": str(row.pdf_assignment_source) if hasattr(row, "pdf_assignment_source") else "unknown",
                "success": None if not hasattr(row, "success") or pd.isna(row.success) else int(row.success),
                "nfev": None if not hasattr(row, "nfev") or pd.isna(row.nfev) else int(row.nfev),
                "best_chi2dN": None if not hasattr(row, "best_chi2dN") or pd.isna(row.best_chi2dN) else float(row.best_chi2dN),
            }
        )

    elapsed_seconds = time.perf_counter() - start_time

    chunk_payload = {
        "metadata": {
            "fit_name": config["fit_name"],
            "chunk_csv": str(chunk_csv),
            "output_path": str(output_path),
            "table_name": table_name,
            "elapsed_seconds": float(elapsed_seconds),
            "n_rows": int(len(chunk_df)),
            "curve_keys": curve_keys,
            "kt_grid": kt_plot,
        },
        "replica_info": pd.DataFrame(assignment_rows),
        "kt_samples": {key: np.stack(kt_samples[key], axis=0) for key in curve_keys},
    }

    with output_path.open("wb") as handle:
        pickle.dump(chunk_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Wrote chunk output to {output_path} with {len(chunk_df)} rows "
        f"in {elapsed_seconds:.1f} s"
    )


if __name__ == "__main__":
    main()
