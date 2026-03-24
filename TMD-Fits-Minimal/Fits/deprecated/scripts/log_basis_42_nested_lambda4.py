from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, CandidateSpec
from art23_family_search import Art23FamilyFitSession
from parameter_significance_0_2 import NestedFit, parse_array, parse_struct_fields


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "log_basis_42_nested_prune_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_CARD = CARDS_DIR / "BroadBump42XbarQuadLogPair.jl"
BASE_RESULT = FITS_DIR / "log_basis_42_results" / "xbar_quad_logpair_bump.json"


def make_spec() -> CandidateSpec:
    card_text = BASE_CARD.read_text(encoding="utf-8")
    best = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    best_full = [float(x) for x in best["best"]["full_params"]]
    return CandidateSpec(
        name="xbar_quad_logpair_bump_lambda4_zero_nested",
        fit_name="BroadBump42XbarQuadLogPair",
        np_name=card_text.split('const NP_name = "', 1)[1].split('"', 1)[0],
        param_names=parse_struct_fields(card_text),
        initial_params=best_full,
        bounds=[(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")],
        frozen_indices=[int(x) for x in parse_array(card_text, "frozen_indices")],
        kernel_variant="nested_prune",
    )


def render_card(spec: CandidateSpec, full_params: list[float]) -> str:
    struct_fields = "\n".join(f"    {name}::Float32" for name in spec.param_names)
    init_vals = ", ".join(f"{x:.10g}" for x in full_params)
    bounds_vals = ",\n    ".join(f"({lo:.10g}, {hi:.10g})" for lo, hi in spec.bounds)
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

frozen_indices = [3, 4]

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


def main() -> None:
    spec = make_spec()
    session = Art23FamilyFitSession(spec)
    best_full = np.asarray(spec.initial_params, dtype=float)
    nested = NestedFit(session, best_full, {"lambda4": 0.0})
    fit_info = nested.solve(maxfun=180)
    best_nested = nested.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))
    full_params = [float(x) for x in best_nested["full_params"]]

    out = {
        "candidate": "xbar_quad_logpair_bump_lambda4_zero_nested",
        "fit_name": "BroadBump42XbarQuadLogPairNoLambda4",
        "fit": fit_info,
        "best": best_nested,
    }
    (RESULTS_DIR / "xbar_quad_logpair_bump_lambda4_zero_nested.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    (CARDS_DIR / "BroadBump42XbarQuadLogPairNoLambda4.jl").write_text(
        render_card(spec, full_params), encoding="utf-8"
    )

    ref = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    ref_metrics = ref["best"]["metrics"]
    new_metrics = best_nested["metrics"]
    pd.DataFrame(
        [
            {
                "reference_candidate": "xbar_quad_logpair_bump",
                "new_candidate": "xbar_quad_logpair_bump_lambda4_zero_nested",
                "chi2dN_total_ref": ref_metrics["chi2dN_total"],
                "chi2dN_total_new": new_metrics["chi2dN_total"],
                "delta_chi2dN_total": new_metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
                "highE_absdev_ref": ref_metrics["highE_mean_absdev_first3"],
                "highE_absdev_new": new_metrics["highE_mean_absdev_first3"],
                "delta_absdev": new_metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
                "highE_shortfall_ref": ref_metrics["highE_mean_shortfall_first3"],
                "highE_shortfall_new": new_metrics["highE_mean_shortfall_first3"],
                "delta_shortfall": new_metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
                "fit_evals": fit_info["nf"],
                "fit_flag": fit_info["flag"],
            }
        ]
    ).to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("Saved BroadBump42XbarQuadLogPairNoLambda4.jl")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
