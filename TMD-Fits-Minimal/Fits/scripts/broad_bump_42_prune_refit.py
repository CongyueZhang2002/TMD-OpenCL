from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from art23_family_search import Art23FamilyFitSession
from auto_np_search import CARDS_DIR, FITS_DIR, PYTHON, CandidateSpec
from _paths import RESULTS_ROOT


RESULTS_DIR = RESULTS_ROOT / "broad_bump_42_prune_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = RESULTS_ROOT / "broad_bump_42_results" / "loggauss_w060_alpha1.json"

PARAM_NAMES = [
    "lambda1",
    "lambda2",
    "lambda3",
    "lambda4",
    "alpha",
    "logx0",
    "sigx",
    "amp",
    "BNP",
    "c0",
    "c1",
]


def _best_full_params(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(map(float, data["best"]["full_params"]))


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


def candidate_specs() -> list[CandidateSpec]:
    base = _best_full_params(BASE_RESULT)
    bounds = [
        (0.02, 8.0),
        (0.02, 8.0),
        (-10.0, 10.0),
        (-0.5, 0.5),
        (0.0, 2.0),
        (-9.210340372, -1.203972804),
        (0.6, 2.5),
        (-3.0, 3.0),
        (0.4, 4.5),
        (0.0, 0.25),
        (0.0, 0.25),
    ]

    l2_zero = base.copy()
    l2_zero[1] = 0.0

    l2_l4_zero = l2_zero.copy()
    l2_l4_zero[3] = 0.0

    return [
        CandidateSpec(
            name="broad_bump_42_alpha1_lambda2_zero",
            fit_name="BroadBump42LogGaussAlpha1NoLambda2",
            np_name="NP-BroadBump42LogGaussW060Alpha1.cl",
            param_names=PARAM_NAMES,
            initial_params=l2_zero,
            bounds=bounds,
            frozen_indices=[1, 4],
            kernel_variant="loggauss",
        ),
        CandidateSpec(
            name="broad_bump_42_alpha1_lambda2_lambda4_zero",
            fit_name="BroadBump42LogGaussAlpha1NoLambda2NoLambda4",
            np_name="NP-BroadBump42LogGaussW060Alpha1.cl",
            param_names=PARAM_NAMES,
            initial_params=l2_l4_zero,
            bounds=bounds,
            frozen_indices=[1, 3, 4],
            kernel_variant="loggauss",
        ),
    ]


def write_candidate_cards() -> dict[str, CandidateSpec]:
    specs = {spec.name: spec for spec in candidate_specs()}
    for spec in specs.values():
        (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
    return specs


def run_candidate(spec: CandidateSpec, maxfun: int) -> Path:
    session = Art23FamilyFitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(fit_info["free_params"])

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
    ref = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]
    rows = [
        {
            "candidate": "broad_bump_42_alpha1_best",
            "fit_name": "BroadBump42LogGaussAlpha1Best",
            "chi2dN_total": ref_metrics["chi2dN_total"],
            "chi2dN_collider": ref_metrics["chi2dN_collider"],
            "chi2dN_fixed_target": ref_metrics["chi2dN_fixed_target"],
            "highE_mean_absdev_first3": ref_metrics["highE_mean_absdev_first3"],
            "highE_mean_shortfall_first3": ref_metrics["highE_mean_shortfall_first3"],
            "delta_total_vs_best": 0.0,
            "delta_absdev_vs_best": 0.0,
            "delta_shortfall_vs_best": 0.0,
            "fit_evals": ref["fit"]["nf"],
            "fit_flag": ref["fit"]["flag"],
        }
    ]

    for name, spec in specs.items():
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["best"]["metrics"]
        row = {
            "candidate": name,
            "fit_name": spec.fit_name,
            "chi2dN_total": metrics["chi2dN_total"],
            "chi2dN_collider": metrics["chi2dN_collider"],
            "chi2dN_fixed_target": metrics["chi2dN_fixed_target"],
            "highE_mean_absdev_first3": metrics["highE_mean_absdev_first3"],
            "highE_mean_shortfall_first3": metrics["highE_mean_shortfall_first3"],
            "delta_total_vs_best": metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
            "delta_absdev_vs_best": metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
            "delta_shortfall_vs_best": metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
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
    specs = write_candidate_cards()
    out_path = run_candidate(specs[candidate], maxfun=maxfun)
    print(f"Wrote {out_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate(candidates: list[str], maxfun: int) -> None:
    specs = write_candidate_cards()
    for name in candidates:
        cmd = [str(PYTHON), str(Path(__file__)), "--candidate", name, "--maxfun", str(maxfun)]
        print(f"\n=== Running {name} ===")
        proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
        print(f"{name}: exit code {proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed for {name}")

    print("\n=== Summary ===")
    df = summarize_results(specs)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[
            "broad_bump_42_alpha1_lambda2_zero",
            "broad_bump_42_alpha1_lambda2_lambda4_zero",
        ],
    )
    parser.add_argument("--maxfun", type=int, default=360)
    args = parser.parse_args()

    if args.candidate:
        run_worker(args.candidate, maxfun=args.maxfun)
        return

    orchestrate(args.candidates, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
