from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, NP_DIR, PYTHON, CandidateSpec, FitSession


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = FITS_DIR / "table_variant_scan_results"
RESULTS_DIR.mkdir(exist_ok=True)

TABLE_VARIANTS = [
    ("MSHT20N3LO-MC-0-0", 0, 0),
    ("MSHT20N3LO-MC-0-2", 0, 2),
    ("MSHT20N3LO-MC-4-2", 4, 2),
    ("MSHT20N3LO-MC-4-4", 4, 4),
]


@dataclass(frozen=True)
class BaseModel:
    name: str
    label: str
    card_path: Path
    kernel_path: Path
    result_json: Path


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
        name="poly_bstar_cslog_loggauss",
        label="Poly-x bstar + CSlog + loggauss",
        card_path=CARDS_DIR / "PolyBstarCSLogLogGauss.jl",
        kernel_path=NP_DIR / "NP-PolyBstarCSLogLogGauss.cl",
        result_json=FITS_DIR / "localized_shape_followup_results" / "poly_bstar_cslog_loggauss.json",
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


def load_best_params(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(x) for x in data["best"]["full_params"]]


def mustar_function_src(n: int) -> str:
    if n == 0:
        return (
            "inline float mustar_func(float b, float Q) {\n"
            "    float mu = bmax / b;\n"
            "    return max(mu, 1.0f);\n"
            "}\n"
        )
    if n == 2:
        return (
            "inline float mustar_func(float b, float Q) {\n"
            "    float t = b / bmax;\n"
            "    float t2 = t * t;\n"
            "    float denom = sqrt(1.0f + t2);\n"
            "    float bstar = b / denom;\n"
            "    float mu = bmax / bstar;\n"
            "    return max(mu, 1.0f);\n"
            "}\n"
        )
    if n == 4:
        return (
            "inline float mustar_func(float b, float Q) {\n"
            "    float t = b / bmax;\n"
            "    float t2 = t * t;\n"
            "    float t4 = t2 * t2;\n"
            "    float denom = sqrt(sqrt(1.0f + t4));\n"
            "    float bstar = b / denom;\n"
            "    float mu = bmax / bstar;\n"
            "    return max(mu, 1.0f);\n"
            "}\n"
        )
    raise ValueError(f"Unsupported n={n}")


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


def replace_mustar_func(kernel_text: str, n: int) -> str:
    pattern = r"(?ms)inline float mustar_func\(float b, float Q\)\s*\{.*?\n\}"
    return re.sub(pattern, mustar_function_src(n).rstrip(), kernel_text, count=1)


def generated_names(model: BaseModel, table_name: str) -> tuple[str, str]:
    suffix = table_name.replace("MSHT20N3LO-MC-", "").replace("-", "_")
    fit_name = f"TableScan_{model.name}_{suffix}"
    np_name = f"NP-{fit_name}.cl"
    return fit_name, np_name


def write_generated_files(model: BaseModel, table_name: str, n: int) -> CandidateSpec:
    fit_name, np_name = generated_names(model, table_name)

    card_text = model.card_path.read_text(encoding="utf-8")
    kernel_text = model.kernel_path.read_text(encoding="utf-8")

    best_params = load_best_params(model.result_json)
    param_names = parse_struct_fields(card_text)
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]

    card_text = replace_np_name(card_text, np_name)
    card_text = replace_table_name(card_text, table_name)
    card_text = replace_bracket_assignment(card_text, "initial_params", best_params)

    kernel_text = replace_mustar_func(kernel_text, n)

    (CARDS_DIR / f"{fit_name}.jl").write_text(card_text, encoding="utf-8")
    (NP_DIR / np_name).write_text(kernel_text, encoding="utf-8")

    return CandidateSpec(
        name=f"{model.name}__{table_name}",
        fit_name=fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=best_params,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="table_scan",
    )


def local_refine(session: FitSession, maxfun: int) -> dict:
    t0 = time.perf_counter()
    start = session.theta0.copy()

    stage1 = session._solve(start, maxfun=maxfun, rhobeg=0.08, rhoend=2e-4, seek_global_minimum=False)
    stage2_budget = max(40, maxfun // 2)
    stage2 = session._solve(np.asarray(stage1.x, dtype=float), maxfun=stage2_budget, rhobeg=0.04, rhoend=1e-6, seek_global_minimum=False)

    elapsed = time.perf_counter() - t0
    params_free = session.denormalize_params(np.asarray(stage2.x, dtype=float))
    return {
        "elapsed_s": elapsed,
        "nf": int(stage1.nf) + int(stage2.nf),
        "flag": int(stage2.flag),
        "free_params": params_free.tolist(),
        "stage1_log10_chi2": float(stage1.f),
        "stage2_log10_chi2": float(stage2.f),
    }


def run_combo(model_name: str, table_name: str, maxfun: int) -> Path:
    model = next(item for item in BASE_MODELS if item.name == model_name)
    _, n, m = next(item for item in TABLE_VARIANTS if item[0] == table_name)
    spec = write_generated_files(model, table_name, n)

    session = FitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = local_refine(session, maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    result = {
        "model": model.name,
        "label": model.label,
        "table_name": table_name,
        "n": n,
        "m": m,
        "fit_name": spec.fit_name,
        "np_name": spec.np_name,
        "param_names": spec.param_names,
        "free_idx": session.free_idx.tolist(),
        "frozen_idx": session.frozen_idx.tolist(),
        "initial": initial_eval,
        "fit": fit_info,
        "best": best_eval,
    }

    out_path = RESULTS_DIR / f"{model.name}__{table_name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        initial_metrics = data["initial"]["metrics"]
        best_metrics = data["best"]["metrics"]
        rows.append(
            {
                "model": data["model"],
                "label": data["label"],
                "table_name": data["table_name"],
                "n": data["n"],
                "m": data["m"],
                "initial_chi2dN_total": initial_metrics["chi2dN_total"],
                "chi2dN_total": best_metrics["chi2dN_total"],
                "chi2dN_collider": best_metrics["chi2dN_collider"],
                "chi2dN_fixed_target": best_metrics["chi2dN_fixed_target"],
                "highE_weighted_chi2dN": best_metrics["highE_weighted_chi2dN"],
                "highE_mean_absdev_first3": best_metrics["highE_mean_absdev_first3"],
                "highE_mean_shortfall_first3": best_metrics["highE_mean_shortfall_first3"],
                "cms_highmass_mean_signed_first3": best_metrics["cms_highmass_mean_signed_first3"],
                "zlike_mean_signed_first3": best_metrics["zlike_mean_signed_first3"],
                "delta_chi2_vs_initial": best_metrics["chi2dN_total"] - initial_metrics["chi2dN_total"],
                "delta_absdev_vs_initial": best_metrics["highE_mean_absdev_first3"] - initial_metrics["highE_mean_absdev_first3"],
                "fit_evals": data["fit"]["nf"],
                "fit_elapsed_s": data["fit"]["elapsed_s"],
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "highE_mean_absdev_first3", "chi2dN_total"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    rank_rows = []
    for metric in ["chi2dN_total", "highE_mean_absdev_first3", "highE_mean_shortfall_first3"]:
        rank_col = f"rank_{metric}"
        df[rank_col] = df.groupby("model")[metric].rank(method="dense")

    agg = (
        df.groupby(["table_name", "n", "m"], as_index=False)[
            ["rank_chi2dN_total", "rank_highE_mean_absdev_first3", "rank_highE_mean_shortfall_first3"]
        ]
        .mean()
        .sort_values(["rank_highE_mean_absdev_first3", "rank_chi2dN_total", "rank_highE_mean_shortfall_first3"])
        .reset_index(drop=True)
    )
    agg.to_csv(RESULTS_DIR / "aggregate_ranks.csv", index=False)

    lines = [
        "# Table Variant Scan",
        "",
        "Scanned models:",
        "",
    ]
    for model in BASE_MODELS:
        lines.append(f"- `{model.name}`: {model.label}")
    lines.extend(
        [
            "",
            "Tables scanned:",
            "",
        ]
    )
    for table_name, n, m in TABLE_VARIANTS:
        lines.append(f"- `{table_name}` with `n={n}`, `m={m}`")
    lines.extend(
        [
            "",
            "Per-model best table by metric:",
            "",
        ]
    )
    for model_name in df["model"].drop_duplicates():
        subset = df[df["model"] == model_name]
        best_abs = subset.sort_values(["highE_mean_absdev_first3", "chi2dN_total"]).iloc[0]
        best_short = subset.sort_values(["highE_mean_shortfall_first3", "chi2dN_total"]).iloc[0]
        best_chi2 = subset.sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).iloc[0]
        lines.append(
            f"- `{model_name}`: best `chi2` = `{best_chi2['table_name']}` ({best_chi2['chi2dN_total']:.6f}), "
            f"best `absdev` = `{best_abs['table_name']}` ({best_abs['highE_mean_absdev_first3']:.6f}), "
            f"best `shortfall` = `{best_short['table_name']}` ({best_short['highE_mean_shortfall_first3']:.6f})"
        )
    lines.extend(
        [
            "",
            "Aggregate mean ranks across the scanned models are in `aggregate_ranks.csv`.",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nPer-model results:")
    print(df.to_string(index=False))
    print("\nAggregate mean ranks:")
    print(agg.to_string(index=False))


def cleanup_generated_files() -> None:
    prefixes = [f"TableScan_{model.name}_" for model in BASE_MODELS]
    for path in CARDS_DIR.glob("TableScan_*.jl"):
        if any(path.name.startswith(prefix) for prefix in prefixes):
            path.unlink(missing_ok=True)
    for path in NP_DIR.glob("NP-TableScan_*.cl"):
        if any(path.name.startswith(f"NP-{prefix}") for prefix in prefixes):
            path.unlink(missing_ok=True)


def orchestrate(maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for model in BASE_MODELS:
            for table_name, _, _ in TABLE_VARIANTS:
                cmd = [
                    str(PYTHON),
                    str(Path(__file__)),
                    "--combo",
                    model.name,
                    table_name,
                    "--maxfun",
                    str(maxfun),
                ]
                print(f"\n=== Running {model.name} on {table_name} ===")
                proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
                print(f"exit code {proc.returncode}")
        print("\n=== Summary ===")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", nargs=2, metavar=("MODEL", "TABLE"))
    parser.add_argument("--maxfun", type=int, default=90)
    args = parser.parse_args()

    if args.combo:
        model_name, table_name = args.combo
        out_path = run_combo(model_name, table_name, maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(maxfun=args.maxfun)


if __name__ == "__main__":
    main()
