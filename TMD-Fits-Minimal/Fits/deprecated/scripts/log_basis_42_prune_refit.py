from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from art23_family_search import Art23FamilyFitSession
from auto_np_search import CARDS_DIR, FITS_DIR, CandidateSpec


RESULTS_DIR = FITS_DIR / "log_basis_42_prune_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_RESULT = FITS_DIR / "log_basis_42_results" / "xbar_quad_logpair_bump.json"


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


def candidate_spec() -> CandidateSpec:
    base = _best_full_params(BASE_RESULT)
    base[3] = 0.0
    return CandidateSpec(
        name="xbar_quad_logpair_bump_lambda4_zero",
        fit_name="BroadBump42XbarQuadLogPairNoLambda4",
        np_name="NP-BroadBump42XbarQuadLogPair.cl",
        param_names=["lambda1", "lambda2", "lambda3", "lambda4", "alpha", "logx0", "sigx", "amp", "BNP", "c0", "c1"],
        initial_params=base,
        bounds=[
            (0.02, 8.0),
            (-10.0, 10.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (0.0, 2.0),
            (-9.210340372, -1.203972804),
            (0.6, 2.5),
            (-3.0, 3.0),
            (0.4, 4.5),
            (0.0, 0.25),
            (0.0, 0.25),
        ],
        frozen_indices=[3, 4],
        kernel_variant="xbar_quad_logpair",
    )


def run_candidate(spec: CandidateSpec, maxfun: int) -> Path:
    (CARDS_DIR / f"{spec.fit_name}.jl").write_text(_render_card(spec), encoding="utf-8")
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


def summarize() -> None:
    ref = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    new = json.loads((RESULTS_DIR / "xbar_quad_logpair_bump_lambda4_zero.json").read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]
    new_metrics = new["best"]["metrics"]
    row = {
        "reference_candidate": "xbar_quad_logpair_bump",
        "new_candidate": "xbar_quad_logpair_bump_lambda4_zero",
        "chi2dN_total_ref": ref_metrics["chi2dN_total"],
        "chi2dN_total_new": new_metrics["chi2dN_total"],
        "delta_chi2dN_total": new_metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
        "highE_absdev_ref": ref_metrics["highE_mean_absdev_first3"],
        "highE_absdev_new": new_metrics["highE_mean_absdev_first3"],
        "delta_absdev": new_metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
        "highE_shortfall_ref": ref_metrics["highE_mean_shortfall_first3"],
        "highE_shortfall_new": new_metrics["highE_mean_shortfall_first3"],
        "delta_shortfall": new_metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
        "fit_evals": new["fit"]["nf"],
        "fit_flag": new["fit"]["flag"],
    }
    pd.DataFrame([row]).to_csv(RESULTS_DIR / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxfun", type=int, default=360)
    args = parser.parse_args()

    spec = candidate_spec()
    out_path = run_candidate(spec, maxfun=args.maxfun)
    summarize()
    print(f"Wrote {out_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
