from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec, FitSession
from scan_table_variants import mustar_function_src


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "pruned_model_refits_0_2_results"
RESULTS_DIR.mkdir(exist_ok=True)
TARGET_TABLE = "MSHT20N3LO-MC-0-2"


@dataclass(frozen=True)
class PruneCase:
    name: str
    label: str
    base_model: str
    card_path: Path
    kernel_path: Path
    base_result_path: Path
    fixed_map: dict[str, float]


CASES = [
    PruneCase(
        name="baseline_drop_a3_b1",
        label="0112 baseline with a3=0, b1=0",
        base_model="baseline_unfrozen",
        card_path=CARDS_DIR / "AutoBaselineUnfrozen.jl",
        kernel_path=NP_DIR / "NP-AutoBaselineUnfrozen.cl",
        base_result_path=FITS_DIR / "power_table_refit_results" / "baseline_unfrozen__MSHT20N3LO-MC-0-2.json",
        fixed_map={"a3": 0.0, "b1": 0.0},
    ),
    PruneCase(
        name="poly_bstar_cslog_alpha1",
        label="Poly-x bstar + CSlog with alpha=1",
        base_model="poly_bstar_cslog",
        card_path=CARDS_DIR / "Art23FamilyMuPolyBstarCSLog.jl",
        kernel_path=NP_DIR / "NP-Art23FamilyMuPolyBstarCSLog.cl",
        base_result_path=FITS_DIR / "power_table_refit_results" / "poly_bstar_cslog__MSHT20N3LO-MC-0-2.json",
        fixed_map={"alpha": 1.0},
    ),
    PruneCase(
        name="poly_bstar_cslog_loggauss_lambda3_0",
        label="Poly-x bstar + CSlog + loggauss with lambda3=0",
        base_model="poly_bstar_cslog_loggauss",
        card_path=CARDS_DIR / "PolyBstarCSLogLogGauss.jl",
        kernel_path=NP_DIR / "NP-PolyBstarCSLogLogGauss.cl",
        base_result_path=FITS_DIR / "power_table_refit_results" / "poly_bstar_cslog_loggauss__MSHT20N3LO-MC-0-2.json",
        fixed_map={"lambda3": 0.0},
    ),
    PruneCase(
        name="poly_bstar_cslog_loggauss_lambda3_0_alpha1",
        label="Poly-x bstar + CSlog + loggauss with lambda3=0, alpha=1",
        base_model="poly_bstar_cslog_loggauss",
        card_path=CARDS_DIR / "PolyBstarCSLogLogGauss.jl",
        kernel_path=NP_DIR / "NP-PolyBstarCSLogLogGauss.cl",
        base_result_path=FITS_DIR / "power_table_refit_results" / "poly_bstar_cslog_loggauss__MSHT20N3LO-MC-0-2.json",
        fixed_map={"lambda3": 0.0, "alpha": 1.0},
    ),
]


def parse_struct_fields(card_text: str) -> list[str]:
    match = re.search(r"struct\s+Params_Struct(.*?)end", card_text, re.S)
    if not match:
        raise RuntimeError("Could not find Params_Struct block")
    fields = []
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


def replace_bracket_assignment(text: str, name: str, values: list[float | int]) -> str:
    rendered = ", ".join(f"{float(v):.12g}" if isinstance(v, (float, int, np.floating, np.integer)) else str(v) for v in values)
    pattern = rf"(?ms)^([ \t]*(?!#){re.escape(name)}\s*=\s*)\[(.*?)\]"
    return re.sub(pattern, rf"\1[{rendered}]", text, count=1)


def replace_scalar_assignment(text: str, name: str, value: str) -> str:
    pattern = rf'(?m)^([ \t]*(?!#){re.escape(name)}\s*=\s*)".*?"'
    return re.sub(pattern, rf'\1"{value}"', text, count=1)


def replace_mustar_func(kernel_text: str, n: int) -> str:
    pattern = r"(?ms)inline float mustar_func\(float b, float Q\)\s*\{.*?\n\}"
    return re.sub(pattern, mustar_function_src(n).rstrip(), kernel_text, count=1)


def load_best_full(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def base_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["best"]["metrics"]


def make_spec(case: PruneCase) -> CandidateSpec:
    card_text = case.card_path.read_text(encoding="utf-8")
    kernel_text = case.kernel_path.read_text(encoding="utf-8")
    param_names = parse_struct_fields(card_text)
    name_to_idx = {name: i for i, name in enumerate(param_names)}

    full = load_best_full(case.base_result_path)
    for name, value in case.fixed_map.items():
        full[name_to_idx[name]] = float(value)

    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]
    frozen = sorted(set(frozen) | {name_to_idx[name] for name in case.fixed_map})

    fit_name = f"Pruned0p2_{case.name}"
    np_name = f"NP-{fit_name}.cl"

    card_text = replace_scalar_assignment(card_text, "const NP_name", np_name)
    card_text = replace_scalar_assignment(card_text, "const table_name", TARGET_TABLE)
    card_text = replace_bracket_assignment(card_text, "initial_params", full)
    card_text = replace_bracket_assignment(card_text, "frozen_indices", frozen)
    kernel_text = replace_mustar_func(kernel_text, 0)

    (CARDS_DIR / f"{fit_name}.jl").write_text(card_text, encoding="utf-8")
    (NP_DIR / np_name).write_text(kernel_text, encoding="utf-8")

    return CandidateSpec(
        name=case.name,
        fit_name=fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=full,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="pruned",
    )


def cleanup_generated_files() -> None:
    for case in CASES:
        fit_name = f"Pruned0p2_{case.name}"
        (CARDS_DIR / f"{fit_name}.jl").unlink(missing_ok=True)
        (NP_DIR / f"NP-{fit_name}.cl").unlink(missing_ok=True)


def run_case(case_name: str, maxfun: int) -> Path:
    case = next(item for item in CASES if item.name == case_name)
    spec = make_spec(case)
    session = FitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "case": case.name,
        "label": case.label,
        "base_model": case.base_model,
        "fixed_map": case.fixed_map,
        "table_name": TARGET_TABLE,
        "fit_name": spec.fit_name,
        "np_name": spec.np_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
        "base_metrics": base_metrics(case.base_result_path),
    }

    out_path = RESULTS_DIR / f"{case.name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        best = data["best"]["metrics"]
        base = data["base_metrics"]
        rows.append(
            {
                "case": data["case"],
                "label": data["label"],
                "base_model": data["base_model"],
                "fixed_map": json.dumps(data["fixed_map"], sort_keys=True),
                "chi2dN_total": best["chi2dN_total"],
                "highE_mean_absdev_first3": best["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": best["highE_mean_shortfall_first3"],
                "delta_total_vs_base": best["chi2dN_total"] - base["chi2dN_total"],
                "delta_absdev_vs_base": best["highE_mean_absdev_first3"] - base["highE_mean_absdev_first3"],
                "delta_shortfall_vs_base": best["highE_mean_shortfall_first3"] - base["highE_mean_shortfall_first3"],
                "fit_evals": data["fit"]["nf"],
                "fit_elapsed_s": data["fit"]["elapsed_s"],
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    lines = [
        "# Pruned Model Refits On Default 0-2",
        "",
        "Cases:",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['case']}` from `{row['base_model']}`: "
            f"`chi2/N={row['chi2dN_total']:.6f}`, "
            f"`dchi2/N={row['delta_total_vs_base']:+.6f}`, "
            f"`dabsdev={row['delta_absdev_vs_base']:+.6f}`, "
            f"`dshortfall={row['delta_shortfall_vs_base']:+.6f}`"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


def orchestrate(maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for case in CASES:
            cmd = [
                str(PYTHON),
                str(Path(__file__)),
                "--case",
                case.name,
                "--maxfun",
                str(maxfun),
            ]
            print(f"\n=== Running {case.name} ===")
            proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
            print(f"exit code {proc.returncode}")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str)
    parser.add_argument("--maxfun", type=int, default=180)
    args = parser.parse_args()

    cleanup_generated_files()
    try:
        if args.case:
            out_path = run_case(args.case, maxfun=args.maxfun)
            print(f"Wrote {out_path}")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        else:
            orchestrate(maxfun=args.maxfun)
    finally:
        cleanup_generated_files()


if __name__ == "__main__":
    main()
