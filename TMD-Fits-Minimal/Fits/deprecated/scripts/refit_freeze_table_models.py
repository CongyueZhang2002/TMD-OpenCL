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


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "freeze_table_refit_results"
RESULTS_DIR.mkdir(exist_ok=True)

FREEZE_TABLE = "MSHT20N3LO-MC-freeze"


@dataclass(frozen=True)
class BaseModel:
    name: str
    label: str
    card_path: Path
    kernel_path: Path
    result_json: Path | None = None


BASE_MODELS = [
    BaseModel(
        name="baseline_unfrozen",
        label="0112 baseline",
        card_path=CARDS_DIR / "AutoBaselineUnfrozen.jl",
        kernel_path=NP_DIR / "NP-AutoBaselineUnfrozen.cl",
        result_json=FITS_DIR / "art23_family_results" / "baseline_unfrozen.json",
    ),
    BaseModel(
        name="poly_bstar_cslog",
        label="Poly-x bstar + CSlog",
        card_path=CARDS_DIR / "Art23FamilyMuPolyBstarCSLog.jl",
        kernel_path=NP_DIR / "NP-Art23FamilyMuPolyBstarCSLog.cl",
        result_json=FITS_DIR / "art23_family_results" / "art23_mu_poly_bstar_cslog.json",
    ),
    BaseModel(
        name="reduced_loggauss_powerseed",
        label="Reduced loggauss powerseed",
        card_path=CARDS_DIR / "ReducedLogGaussPowerSeed.jl",
        kernel_path=NP_DIR / "NP-ReducedLogGaussPowerSeed.cl",
        result_json=FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json",
    ),
    BaseModel(
        name="xb_bump_logb_nobump",
        label="XB logb no-bump",
        card_path=CARDS_DIR / "XBBumpLogBNoBump.jl",
        kernel_path=NP_DIR / "NP-XBBumpLogBNoBump.cl",
        result_json=FITS_DIR / "xb_bump_logb_nobump_results" / "xb_bump_logb_nobump.json",
    ),
]


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


def parse_initial_params_from_card(card_text: str) -> list[float]:
    return [float(x) for x in parse_array(card_text, "initial_params")]


def load_best_params(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def replace_bracket_assignment(text: str, name: str, values: list[float]) -> str:
    rendered = ", ".join(f"{v:.12g}" for v in values)
    pattern = rf"(?ms)^([ \t]*(?!#){re.escape(name)}\s*=\s*)\[(.*?)\]"
    return re.sub(pattern, rf"\1[{rendered}]", text, count=1)


def replace_scalar_assignment(text: str, name: str, value: str) -> str:
    pattern = rf'(?m)^([ \t]*(?!#){re.escape(name)}\s*=\s*)".*?"'
    return re.sub(pattern, rf'\1"{value}"', text, count=1)


def replace_np_name(text: str, np_name: str) -> str:
    return replace_scalar_assignment(text, "const NP_name", np_name)


def replace_table_name(text: str, table_name: str) -> str:
    return replace_scalar_assignment(text, "const table_name", table_name)


def freeze_mustar_function_src() -> str:
    return (
        "inline float mustar_func(float b, float Q) {\n"
        "    float mu = bmax / b;\n"
        "    return max(mu, 1.0f);\n"
        "}\n"
    )


def replace_mustar_func(kernel_text: str) -> str:
    pattern = r"(?ms)inline float mustar_func\(float b, float Q\)\s*\{.*?\n\}"
    if not re.search(pattern, kernel_text):
        raise RuntimeError("Could not find mustar_func in kernel")
    return re.sub(pattern, freeze_mustar_function_src().rstrip(), kernel_text, count=1)


def generated_names(model: BaseModel) -> tuple[str, str]:
    fit_name = f"FreezeRefit_{model.name}"
    np_name = f"NP-{fit_name}.cl"
    return fit_name, np_name


def start_params_for(model: BaseModel, card_text: str) -> list[float]:
    if model.result_json is not None and model.result_json.exists():
        return load_best_params(model.result_json)
    return parse_initial_params_from_card(card_text)


def write_generated_files(model: BaseModel) -> CandidateSpec:
    fit_name, np_name = generated_names(model)

    card_text = model.card_path.read_text(encoding="utf-8")
    kernel_text = model.kernel_path.read_text(encoding="utf-8")

    start_params = start_params_for(model, card_text)
    param_names = parse_struct_fields(card_text)
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]

    card_text = replace_np_name(card_text, np_name)
    card_text = replace_table_name(card_text, FREEZE_TABLE)
    card_text = replace_bracket_assignment(card_text, "initial_params", start_params)
    kernel_text = replace_mustar_func(kernel_text)

    (CARDS_DIR / f"{fit_name}.jl").write_text(card_text, encoding="utf-8")
    (NP_DIR / np_name).write_text(kernel_text, encoding="utf-8")

    return CandidateSpec(
        name=f"{model.name}__freeze",
        fit_name=fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=start_params,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="freeze_refit",
    )


def run_model(model_name: str, maxfun: int) -> Path:
    model = next(item for item in BASE_MODELS if item.name == model_name)
    spec = write_generated_files(model)

    session = FitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    reference_best = None
    if model.result_json is not None and model.result_json.exists():
        reference_best = json.loads(model.result_json.read_text(encoding="utf-8")).get("best")

    result = {
        "model": model.name,
        "label": model.label,
        "table_name": FREEZE_TABLE,
        "freeze_mustar": "1.0",
        "fit_name": spec.fit_name,
        "np_name": spec.np_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
        "reference_best_0p2": reference_best,
    }

    out_path = RESULTS_DIR / f"{model.name}__freeze.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*__freeze.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        initial_metrics = data["initial"]["metrics"]
        best_metrics = data["best"]["metrics"]
        ref = data.get("reference_best_0p2")
        ref_metrics = ref["metrics"] if ref is not None else None

        row = {
            "model": data["model"],
            "label": data["label"],
            "table_name": data["table_name"],
            "initial_chi2dN_total": initial_metrics["chi2dN_total"],
            "freeze_chi2dN_total": best_metrics["chi2dN_total"],
            "freeze_chi2dN_collider": best_metrics["chi2dN_collider"],
            "freeze_chi2dN_fixed_target": best_metrics["chi2dN_fixed_target"],
            "freeze_highE_weighted_chi2dN": best_metrics["highE_weighted_chi2dN"],
            "freeze_highE_mean_absdev_first3": best_metrics["highE_mean_absdev_first3"],
            "freeze_highE_mean_shortfall_first3": best_metrics["highE_mean_shortfall_first3"],
            "freeze_cms_highmass_mean_signed_first3": best_metrics["cms_highmass_mean_signed_first3"],
            "freeze_zlike_mean_signed_first3": best_metrics["zlike_mean_signed_first3"],
            "fit_evals": data["fit"]["nf"],
            "fit_elapsed_s": data["fit"]["elapsed_s"],
        }

        if ref_metrics is not None:
            row.update(
                {
                    "ref_0p2_chi2dN_total": ref_metrics["chi2dN_total"],
                    "ref_0p2_highE_mean_absdev_first3": ref_metrics["highE_mean_absdev_first3"],
                    "ref_0p2_highE_mean_shortfall_first3": ref_metrics["highE_mean_shortfall_first3"],
                    "delta_chi2_freeze_minus_0p2": best_metrics["chi2dN_total"] - ref_metrics["chi2dN_total"],
                    "delta_absdev_freeze_minus_0p2": best_metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"],
                    "delta_shortfall_freeze_minus_0p2": best_metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"],
                }
            )

        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["freeze_highE_mean_absdev_first3", "freeze_chi2dN_total"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    lines = [
        "# Freeze Table Refit Results",
        "",
        f"Table: `{FREEZE_TABLE}`",
        "",
        "Assumption used in the generated kernels:",
        "",
        "- `mustar_func(b, Q) = max(bmax / b, 1.0f)`, matching the latest requested freeze-table check.",
        "",
        "Models scanned:",
        "",
    ]
    for model in BASE_MODELS:
        lines.append(f"- `{model.name}`: {model.label}")
    lines.extend(["", "Results:", ""])

    for _, row in df.iterrows():
        parts = [
            f"`{row['model']}`: freeze chi2/N = `{row['freeze_chi2dN_total']:.6f}`",
            f"absdev = `{row['freeze_highE_mean_absdev_first3']:.6f}`",
            f"shortfall = `{row['freeze_highE_mean_shortfall_first3']:.6f}`",
        ]
        if "ref_0p2_chi2dN_total" in row and not pd.isna(row["ref_0p2_chi2dN_total"]):
            parts.append(f"delta vs 0-2 chi2 = `{row['delta_chi2_freeze_minus_0p2']:+.6f}`")
            parts.append(f"delta vs 0-2 absdev = `{row['delta_absdev_freeze_minus_0p2']:+.6f}`")
            parts.append(f"delta vs 0-2 shortfall = `{row['delta_shortfall_freeze_minus_0p2']:+.6f}`")
        lines.append("- " + ", ".join(parts))

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


def cleanup_generated_files() -> None:
    prefixes = [f"FreezeRefit_{model.name}" for model in BASE_MODELS]
    for path in CARDS_DIR.glob("FreezeRefit_*.jl"):
        if any(path.stem.startswith(prefix) for prefix in prefixes):
            path.unlink(missing_ok=True)
    for path in NP_DIR.glob("NP-FreezeRefit_*.cl"):
        if any(path.stem.startswith(f"NP-{prefix}") for prefix in prefixes):
            path.unlink(missing_ok=True)


def orchestrate(model_names: list[str], maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for model_name in model_names:
            cmd = [
                str(PYTHON),
                str(Path(__file__)),
                "--model",
                model_name,
                "--maxfun",
                str(maxfun),
            ]
            print(f"\n=== Freeze refit {model_name} on {FREEZE_TABLE} ===")
            proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
            print(f"exit code {proc.returncode}")
        print("\n=== Summary ===")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="*", default=[model.name for model in BASE_MODELS])
    parser.add_argument("--maxfun", type=int, default=180)
    args = parser.parse_args()

    if args.model:
        out_path = run_model(args.model, maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(args.models, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
