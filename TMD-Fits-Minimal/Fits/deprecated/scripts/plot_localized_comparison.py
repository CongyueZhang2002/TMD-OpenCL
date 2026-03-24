from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import plot_model_comparison as pmc
from auto_np_search import FITS_DIR, PYTHON, FitSession, candidate_specs as auto_candidate_specs, write_candidate_files as write_auto_candidate_files
from art23_family_search import RESULTS_DIR as FAMILY_RESULTS_DIR, write_family_candidate_files
from art23_family_refine import write_refine_candidate_files
from localized_shape_followup import RESULTS_DIR as LOCAL_RESULTS_DIR, write_followup_candidate_files as write_local_candidate_files
from poly_cs_power_followup import RESULTS_DIR as POWER_RESULTS_DIR, write_followup_candidate_files as write_power_candidate_files


OUT_DIR = FITS_DIR / "model_comparison_plots" / "localized_comparison"

SHORTLIST = {
    "baseline_unfrozen": {
        "result_path": FAMILY_RESULTS_DIR / "baseline_unfrozen.json",
        "label": "0112 baseline",
        "color": "tab:orange",
        "linestyle": "-",
    },
    "art23_mu_poly_bstar_cslog": {
        "result_path": FAMILY_RESULTS_DIR / "art23_mu_poly_bstar_cslog.json",
        "label": "Poly-x bstar + CSlog",
        "color": "tab:red",
        "linestyle": ":",
    },
    "poly_bstar_cslog_loggauss": {
        "result_path": LOCAL_RESULTS_DIR / "poly_bstar_cslog_loggauss.json",
        "label": "Poly-x bstar + CSlog + loggauss",
        "color": "tab:green",
        "linestyle": "-.",
    },
    "poly_cslog2_cspower": {
        "result_path": POWER_RESULTS_DIR / "poly_cslog2_cspower.json",
        "label": "Poly-x + CSlog2 + powerCS",
        "color": "tab:blue",
        "linestyle": "--",
    },
    "poly_bstar_cslog_cspower": {
        "result_path": POWER_RESULTS_DIR / "poly_bstar_cslog_cspower.json",
        "label": "Poly-x bstar + CSlog + powerCS",
        "color": "tab:purple",
        "linestyle": (0, (3, 1, 1, 1)),
    },
}


def load_specs() -> dict[str, object]:
    write_auto_candidate_files()
    specs = {}
    specs.update({spec.name: spec for spec in auto_candidate_specs() if spec.name == "baseline_unfrozen"})
    specs.update(write_family_candidate_files())
    specs.update(write_refine_candidate_files())
    specs.update(write_power_candidate_files())
    specs.update(write_local_candidate_files())
    return specs


def export_candidate_rows(candidate: str, specs: dict[str, object], files: list[str], out_dir: Path) -> None:
    spec = specs[candidate]
    session = FitSession(spec)
    result = json.loads(Path(SHORTLIST[candidate]["result_path"]).read_text(encoding="utf-8"))
    full_params = pmc.reconstruct_full_params(result)

    predictions, compute_s = session._predict(full_params)
    _, chi2_list, n_list = session.get_chi2(predictions)
    metrics = session.summary_metrics(predictions, chi2_list, compute_s)

    rows: list[dict[str, float | int | str]] = []
    for file_name in files:
        if file_name not in predictions:
            continue

        df_data = session.data_list[file_name].reset_index(drop=True)
        pred = np.asarray(predictions[file_name], dtype=float)
        data = df_data["xsec"].to_numpy(dtype=float)
        qT = df_data["qT_mean"].to_numpy(dtype=float)
        ratio = pred / data

        matrix_data = pd.read_csv(session.matrix_root / file_name).to_numpy(dtype=float)
        if session.data_uncertainty_only:
            matrix_total = matrix_data
        else:
            matrix_pdf = pd.read_csv(session.error_sets_root / file_name).to_numpy(dtype=float)
            matrix_total = matrix_data + matrix_pdf

        error_data = np.sqrt(np.diag(matrix_data))
        error_total = np.sqrt(np.diag(matrix_total))
        error_data_ratio = error_data / data
        error_total_ratio = error_total / data

        for i in range(len(qT)):
            rows.append(
                {
                    "candidate": candidate,
                    "label": SHORTLIST[candidate]["label"],
                    "file": file_name,
                    "bin_index": i,
                    "qT": float(qT[i]),
                    "data": float(data[i]),
                    "prediction": float(pred[i]),
                    "ratio": float(ratio[i]),
                    "error_data": float(error_data[i]),
                    "error_total": float(error_total[i]),
                    "error_data_ratio": float(error_data_ratio[i]),
                    "error_total_ratio": float(error_total_ratio[i]),
                    "chi2dN_file": float(chi2_list[file_name]),
                    "N_file": int(n_list[file_name]),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / f"{candidate}_bins.csv", index=False)
    meta = {
        "candidate": candidate,
        "label": SHORTLIST[candidate]["label"],
        "fit_name": result["fit_name"],
        "param_names": result["param_names"],
        "full_params": full_params.tolist(),
        "metrics": metrics,
    }
    (out_dir / f"{candidate}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_worker(candidate: str, files: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    export_candidate_rows(candidate, specs, files, out_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = pmc.build_default_file_list()

    pmc.CANDIDATE_STYLE.update({name: {k: v for k, v in conf.items() if k in {"label", "color", "linestyle"}} for name, conf in SHORTLIST.items()})

    for candidate in SHORTLIST:
        cmd = [
            str(PYTHON),
            str(Path(__file__)),
            "--worker",
            "--candidate",
            candidate,
            "--outdir",
            str(out_dir),
        ]
        print(f"\n=== Exporting {candidate} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{candidate}: exit code {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {candidate}")

    summary = pmc.summarize(out_dir, list(SHORTLIST))
    print("\nSummary:")
    print(summary.to_string(index=False))

    pmc.plot_combined(
        out_dir,
        list(SHORTLIST),
        files,
        write_individual=False,
        annotate_chi2=False,
        grid_filename="all_experiments_grid.png",
    )
    pmc.plot_combined(
        out_dir,
        list(SHORTLIST),
        pmc.DEFAULT_FILES,
        write_individual=False,
        annotate_chi2=False,
        grid_filename="high_energy_grid.png",
    )

    rows = []
    for candidate in SHORTLIST:
        meta = json.loads((out_dir / f"{candidate}_meta.json").read_text(encoding="utf-8"))
        metrics = meta["metrics"]
        rows.append(
            {
                "candidate": candidate,
                "label": meta["label"],
                "chi2dN_total": metrics["chi2dN_total"],
                "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "localized_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", type=str)
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    if args.worker:
        if not args.candidate:
            raise ValueError("--worker requires --candidate")
        run_worker(args.candidate, pmc.build_default_file_list(), out_dir)
        return

    orchestrate(out_dir)


if __name__ == "__main__":
    main()
