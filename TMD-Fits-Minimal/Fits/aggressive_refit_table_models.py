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
RESULTS_DIR = FITS_DIR / "aggressive_table_refit_results"
RESULTS_DIR.mkdir(exist_ok=True)

TABLES = [
    ("MSHT20N3LO-MC-0-2", 0, 2),
    ("MSHT20N3LO-MC-4-2", 4, 2),
    ("MSHT20N3LO-MC-freeze", 0, -1),
]


@dataclass(frozen=True)
class BaseModel:
    name: str
    label: str
    card_path: Path
    kernel_path: Path
    result_json: Path


MODELS = [
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
    BaseModel(
        name="reduced_loggauss_powerseed",
        label="Reduced loggauss powerseed",
        card_path=CARDS_DIR / "ReducedLogGaussPowerSeed.jl",
        kernel_path=NP_DIR / "NP-ReducedLogGaussPowerSeed.cl",
        result_json=FITS_DIR / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json",
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


def replace_bracket_assignment(text: str, name: str, values: list[float]) -> str:
    rendered = ", ".join(f"{v:.12g}" for v in values)
    pattern = rf"(?ms)^([ \t]*(?!#){re.escape(name)}\s*=\s*)\[(.*?)\]"
    return re.sub(pattern, rf"\1[{rendered}]", text, count=1)


def replace_scalar_assignment(text: str, name: str, value: str) -> str:
    pattern = rf'(?m)^([ \t]*(?!#){re.escape(name)}\s*=\s*)".*?"'
    return re.sub(pattern, rf'\1"{value}"', text, count=1)


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


def replace_np_name(text: str, np_name: str) -> str:
    return replace_scalar_assignment(text, "const NP_name", np_name)


def replace_table_name(text: str, table_name: str) -> str:
    return replace_scalar_assignment(text, "const table_name", table_name)


def replace_mustar_func(kernel_text: str, n: int) -> str:
    pattern = r"(?ms)inline float mustar_func\(float b, float Q\)\s*\{.*?\n\}"
    if not re.search(pattern, kernel_text):
        raise RuntimeError("Could not find mustar_func in kernel")
    return re.sub(pattern, mustar_function_src(n).rstrip(), kernel_text, count=1)


def cleanup_generated_files() -> None:
    for path in CARDS_DIR.glob("AggressiveTableRefit_*.jl"):
        path.unlink(missing_ok=True)
    for path in NP_DIR.glob("NP-AggressiveTableRefit_*.cl"):
        path.unlink(missing_ok=True)


def reference_json_for(model: BaseModel, table_name: str) -> Path | None:
    if table_name == "MSHT20N3LO-MC-freeze":
        path = FITS_DIR / "freeze_table_refit_results" / f"{model.name}__freeze.json"
        if path.exists():
            return path
        return model.result_json if model.result_json.exists() else None
    if table_name in {"MSHT20N3LO-MC-0-2", "MSHT20N3LO-MC-4-2"}:
        table_path = FITS_DIR / "table_variant_refit_results" / f"{model.name}__{table_name}.json"
        if table_path.exists():
            return table_path
    return model.result_json if model.result_json.exists() else None


def write_generated_files(model: BaseModel, table_name: str, n: int) -> CandidateSpec:
    suffix = table_name.replace("MSHT20N3LO-MC-", "").replace("-", "_")
    fit_name = f"AggressiveTableRefit_{model.name}_{suffix}"
    np_name = f"NP-{fit_name}.cl"

    card_text = model.card_path.read_text(encoding="utf-8")
    kernel_text = model.kernel_path.read_text(encoding="utf-8")

    ref_json = reference_json_for(model, table_name)
    if ref_json is None:
        raise RuntimeError(f"No reference start found for {model.name} on {table_name}")

    best_params = load_best_params(ref_json)
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
        name=f"{model.name}__{table_name}__aggressive",
        fit_name=fit_name,
        np_name=np_name,
        param_names=param_names,
        initial_params=best_params,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="aggressive_table_refit",
    )


class AggressiveFitSession(FitSession):
    def candidate_start_points(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = [self.theta0.copy()]
        seed = 20260323 + 31 * sum(ord(c) for c in self.spec.name)
        rng = np.random.default_rng(seed)

        for scale in [0.03, 0.06, 0.10, 0.16]:
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))
            starts.append(np.clip(self.theta0 + rng.normal(0.0, scale, size=self.theta0.shape), 0.0, 1.0))

        focus_names = []
        for pname in ["g2", "bmax_CS", "power_CS", "lambda1", "lambda2", "lambda4", "amp", "BNP", "c0", "c1", "logx0", "sigx"]:
            if pname in self.free_param_names and pname not in focus_names:
                focus_names.append(pname)
            if len(focus_names) >= 4:
                break
        for pname in focus_names:
            idx = self.free_param_names.index(pname)
            for val in [0.12, 0.88]:
                start = self.theta0.copy()
                start[idx] = val
                starts.append(start)

        for _ in range(4):
            starts.append(rng.uniform(0.0, 1.0, size=self.theta0.shape))

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(np.asarray(start, dtype=float), 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(np.asarray(start, dtype=float))
        return unique

    def fit(self, maxfun: int) -> dict:
        t0 = time.perf_counter()
        stage1_budget = max(60, maxfun // 3)
        stage2_budget = maxfun
        stage3_budget = max(90, maxfun // 2)
        polish_budget = max(60, maxfun // 3)

        stage1_results = []
        for i, start in enumerate(self.candidate_start_points()):
            res = self._solve(start, maxfun=stage1_budget, rhobeg=0.18, rhoend=7e-4, seek_global_minimum=True)
            stage1_results.append(
                {
                    "start_id": i,
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )

        stage1_results.sort(key=lambda item: item["log10_chi2"])
        stage2_results = []
        for item in stage1_results[:4]:
            res = self._solve(item["theta"], maxfun=stage2_budget, rhobeg=0.10, rhoend=8e-5, seek_global_minimum=True)
            stage2_results.append(
                {
                    "start_id": item["start_id"],
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )

        stage2_results.sort(key=lambda item: item["log10_chi2"])
        stage3_results = []
        for item in stage2_results[:2]:
            res = self._solve(item["theta"], maxfun=stage3_budget, rhobeg=0.05, rhoend=2e-6, seek_global_minimum=False)
            stage3_results.append(
                {
                    "start_id": item["start_id"],
                    "theta": np.asarray(res.x, dtype=float),
                    "log10_chi2": float(res.f),
                    "nf": int(res.nf),
                    "flag": int(res.flag),
                }
            )

        stage3_results.sort(key=lambda item: item["log10_chi2"])
        best_stage3 = stage3_results[0]
        res = self._solve(best_stage3["theta"], maxfun=polish_budget, rhobeg=0.025, rhoend=5e-7, seek_global_minimum=False)

        elapsed = time.perf_counter() - t0
        params_free = self.denormalize_params(np.asarray(res.x, dtype=float))
        return {
            "elapsed_s": elapsed,
            "nf": int(sum(item["nf"] for item in stage1_results) + sum(item["nf"] for item in stage2_results) + sum(item["nf"] for item in stage3_results) + int(res.nf)),
            "flag": int(res.flag),
            "free_params": params_free.tolist(),
            "log10_chi2": float(res.f),
            "stage1": [
                {"start_id": item["start_id"], "log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in stage1_results
            ],
            "stage2": [
                {"start_id": item["start_id"], "log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in stage2_results
            ],
            "stage3": [
                {"start_id": item["start_id"], "log10_chi2": item["log10_chi2"], "nf": item["nf"], "flag": item["flag"]}
                for item in stage3_results
            ],
            "polish_nf": int(res.nf),
        }


def run_combo(model_name: str, table_name: str, maxfun: int) -> Path:
    model = next(item for item in MODELS if item.name == model_name)
    _, n, m = next(item for item in TABLES if item[0] == table_name)
    spec = write_generated_files(model, table_name, n)

    session = AggressiveFitSession(spec)
    initial_eval = session.evaluate_free(session.initial_params[session.free_idx])
    fit_info = session.fit(maxfun=maxfun)
    best_eval = session.evaluate_free(np.asarray(fit_info["free_params"], dtype=float))

    ref_json = reference_json_for(model, table_name)
    reference_best = None
    if ref_json is not None and ref_json.exists():
        reference_best = json.loads(ref_json.read_text(encoding="utf-8")).get("best")

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
        "reference_best": reference_best,
    }

    suffix = table_name.replace("MSHT20N3LO-MC-", "")
    out_path = RESULTS_DIR / f"{model.name}__{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        best_metrics = data["best"]["metrics"]
        initial_metrics = data["initial"]["metrics"]
        ref = data.get("reference_best")
        ref_metrics = ref["metrics"] if ref is not None else None

        row = {
            "model": data["model"],
            "label": data["label"],
            "table_name": data["table_name"],
            "n": data["n"],
            "m": data["m"],
            "initial_chi2dN_total": initial_metrics["chi2dN_total"],
            "chi2dN_total": best_metrics["chi2dN_total"],
            "highE_mean_absdev_first3": best_metrics["highE_mean_absdev_first3"],
            "highE_mean_shortfall_first3": best_metrics["highE_mean_shortfall_first3"],
            "cms_highmass_mean_signed_first3": best_metrics["cms_highmass_mean_signed_first3"],
            "zlike_mean_signed_first3": best_metrics["zlike_mean_signed_first3"],
            "fit_evals": data["fit"]["nf"],
            "fit_elapsed_s": data["fit"]["elapsed_s"],
            "stage1_starts": len(data["fit"]["stage1"]),
            "stage2_refines": len(data["fit"]["stage2"]),
            "stage3_refines": len(data["fit"]["stage3"]),
        }
        if ref_metrics is not None:
            row["ref_chi2dN_total"] = ref_metrics["chi2dN_total"]
            row["ref_highE_mean_absdev_first3"] = ref_metrics["highE_mean_absdev_first3"]
            row["ref_highE_mean_shortfall_first3"] = ref_metrics["highE_mean_shortfall_first3"]
            row["delta_chi2_vs_ref"] = best_metrics["chi2dN_total"] - ref_metrics["chi2dN_total"]
            row["delta_absdev_vs_ref"] = best_metrics["highE_mean_absdev_first3"] - ref_metrics["highE_mean_absdev_first3"]
            row["delta_shortfall_vs_ref"] = best_metrics["highE_mean_shortfall_first3"] - ref_metrics["highE_mean_shortfall_first3"]
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["model", "chi2dN_total", "highE_mean_absdev_first3"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    by_model_rows = []
    for model_name in df["model"].drop_duplicates():
        subset = df[df["model"] == model_name]
        best_chi2 = subset.sort_values(["chi2dN_total", "highE_mean_absdev_first3"]).iloc[0]
        best_abs = subset.sort_values(["highE_mean_absdev_first3", "chi2dN_total"]).iloc[0]
        best_short = subset.sort_values(["highE_mean_shortfall_first3", "chi2dN_total"]).iloc[0]
        by_model_rows.append(
            {
                "model": model_name,
                "best_chi2_table": best_chi2["table_name"],
                "best_chi2dN_total": best_chi2["chi2dN_total"],
                "best_absdev_table": best_abs["table_name"],
                "best_absdev": best_abs["highE_mean_absdev_first3"],
                "best_shortfall_table": best_short["table_name"],
                "best_shortfall": best_short["highE_mean_shortfall_first3"],
            }
        )
    pd.DataFrame(by_model_rows).to_csv(RESULTS_DIR / "best_by_model.csv", index=False)

    lines = [
        "# Aggressive Table Refits",
        "",
        "Heavier search than the earlier local/table refits:",
        "- stage 1: many-start global search",
        "- stage 2: top-4 global refinements",
        "- stage 3: top-2 local refinements",
        "- final polish",
        "",
        "Tables scanned:",
    ]
    for table_name, n, m in TABLES:
        lines.append(f"- `{table_name}` with runtime `mustar` based on `n={n}`")
    lines.extend(["", "Best table per model:", ""])
    for row in by_model_rows:
        lines.append(
            f"- `{row['model']}`: best chi2 `{row['best_chi2_table']}` ({row['best_chi2dN_total']:.6f}), "
            f"best absdev `{row['best_absdev_table']}` ({row['best_absdev']:.6f}), "
            f"best shortfall `{row['best_shortfall_table']}` ({row['best_shortfall']:.6f})"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))


def orchestrate(model_names: list[str], table_names: list[str], maxfun: int) -> None:
    cleanup_generated_files()
    try:
        for model_name in model_names:
            for table_name in table_names:
                cmd = [str(PYTHON), str(Path(__file__)), "--combo", model_name, table_name, "--maxfun", str(maxfun)]
                print(f"\n=== Aggressive refit {model_name} on {table_name} ===")
                proc = subprocess.run(cmd, cwd=str(FITS_DIR), check=False)
                print(f"exit code {proc.returncode}")
        print("\n=== Summary ===")
        summarize()
    finally:
        cleanup_generated_files()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", nargs=2, metavar=("MODEL", "TABLE"))
    parser.add_argument("--models", nargs="*", default=[model.name for model in MODELS])
    parser.add_argument("--tables", nargs="*", default=[name for name, _, _ in TABLES])
    parser.add_argument("--maxfun", type=int, default=240)
    args = parser.parse_args()

    if args.combo:
        model_name, table_name = args.combo
        out_path = run_combo(model_name, table_name, maxfun=args.maxfun)
        print(f"Wrote {out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        orchestrate(args.models, args.tables, maxfun=args.maxfun)


if __name__ == "__main__":
    main()
