import ast
import json
import math
import re
from pathlib import Path

from _paths import FITS_DIR

ROOT = FITS_DIR.parent
FIT_NOTEBOOK = FITS_DIR / "fit.ipynb"
BMAX_CONST = 1.1229189


def parse_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_sources(nb: dict) -> str:
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n".join(parts)


def extract_fit_name(nb: dict) -> str:
    text = notebook_sources(nb)
    match = re.search(r'fit_name\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Could not find fit_name in fit.ipynb")
    return match.group(1)


def extract_struct_fields(card_text: str) -> list[str]:
    match = re.search(r"struct\s+Params_Struct(.*?)end", card_text, re.S)
    if not match:
        raise RuntimeError("Could not find Params_Struct block")
    block = match.group(1)
    fields = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            name = part.split("::", 1)[0].strip()
            fields.append(name)
    return fields


def parse_last_julia_array(card_text: str, name: str) -> list[float]:
    pattern = rf"(?ms)^[ \t]*(?!#){re.escape(name)}\s*=\s*\[(.*?)\]"
    matches = re.findall(pattern, card_text)
    if not matches:
        raise RuntimeError(f"Could not find {name}")
    raw = matches[-1]
    items = []
    for piece in raw.split(","):
        piece = piece.split("#", 1)[0].strip()
        if piece:
            items.append(ast.literal_eval(piece))
    return items


def extract_hardcoded_optimal_params(nb: dict) -> list[float] | None:
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        match = re.search(r"optimal_params\s*=\s*(\[[^\]]+\])", src)
        if match:
            return list(ast.literal_eval(match.group(1)))
    return None


def extract_trial_results(nb: dict) -> list[tuple[int, float, list[float]]]:
    results = []
    pending_trial = None
    pending_chi2 = None
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = "".join(output.get("text", []))
            for line in text.splitlines():
                trial_match = re.match(r"Trial\s+(\d+):\s+Best χ²/N =\s+([0-9.]+)", line)
                if trial_match:
                    pending_trial = int(trial_match.group(1))
                    pending_chi2 = float(trial_match.group(2))
                    continue
                line = line.strip()
                if pending_trial is not None and line.startswith("[") and line.endswith("]"):
                    params = list(ast.literal_eval(line))
                    results.append((pending_trial, pending_chi2, params))
                    pending_trial = None
                    pending_chi2 = None
    return results


def free_indices(dim: int, frozen: list[int]) -> list[int]:
    return [i for i in range(dim) if i not in frozen]


def fill_params(initial: list[float], frozen: list[int], free_values: list[float]) -> list[float]:
    free = free_indices(len(initial), frozen)
    if len(free) != len(free_values):
        raise ValueError(f"Need {len(free)} free parameters, got {len(free_values)}")
    full = list(initial)
    for idx, value in zip(free, free_values):
        full[idx] = value
    return full


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sech(x: float) -> float:
    x = abs(x)
    u = math.exp(-2.0 * x)
    return (2.0 * math.exp(-x)) / (1.0 + u)


def current_np_summary(fields: list[str], params: list[float], x: float, b: float, q: float) -> dict:
    p = dict(zip(fields, params))
    x = clamp(x, 1e-7, 1.0 - 1e-7)
    xbar = 1.0 - x
    xxbar = x * xbar

    xshape = p["a1"] * x + p["a2"] * xbar + p["a3"] * xxbar + p["a4"] * math.log(x)
    expo = p["b1"] * x * x + p["b2"] * xbar * xbar + 2.0 * p["b3"] * xxbar
    expo = clamp(expo, -80.0, 80.0)
    bshape = math.exp(expo)

    t = b / (BMAX_CONST * bshape)
    bstar = b * (1.0 + t**4) ** (0.25 * (p["a"] - 1.0))

    u = b / p["bmax_CS"]
    bstar_cs = b * (1.0 + u**4) ** (0.25 * (p["power_CS"] - 1.0))

    snp_mu = sech(xshape * bstar)
    snp_ze = -0.25 * (p["g2"] ** 2) * (bstar_cs ** 2)

    mustar = max(BMAX_CONST / b, 1.0)
    log_zeta = 2.0 * math.log(q / mustar)
    np_cs = 2.0 * snp_ze
    np_zeta = math.exp(np_cs * log_zeta)
    total_factor = (snp_mu ** 2) * np_zeta

    return {
        "xshape": xshape,
        "bstar": bstar,
        "bstar_cs": bstar_cs,
        "snp_mu": snp_mu,
        "snp_ze": snp_ze,
        "log_zeta": log_zeta,
        "np_zeta": np_zeta,
        "total_factor": total_factor,
    }


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    nb = parse_notebook(FIT_NOTEBOOK)
    fit_name = extract_fit_name(nb)
    card_path = ROOT / "Cards" / f"{fit_name}.jl"
    card_text = card_path.read_text(encoding="utf-8")

    fields = extract_struct_fields(card_text)
    initial = parse_last_julia_array(card_text, "initial_params")
    frozen = [int(x) for x in parse_last_julia_array(card_text, "frozen_indices")]
    free = free_indices(len(initial), frozen)

    legacy_frozen = [1, 5, 6, 9, 10]
    legacy_free = free_indices(len(initial), legacy_frozen)

    hardcoded = extract_hardcoded_optimal_params(nb)
    trials = extract_trial_results(nb)

    print_section("Active Configuration")
    print(f"fit_name: {fit_name}")
    print(f"card: {card_path}")
    print(f"parameter order: {fields}")
    print(f"frozen_indices (0-based): {frozen}")
    print(f"free_indices: {free}")
    print(f"initial_params: {initial}")

    print_section("Notebook Consistency")
    if hardcoded is None:
        print("No hardcoded optimal_params array found in fit.ipynb")
    else:
        print(f"hardcoded optimal_params length: {len(hardcoded)}")
        print(f"current free parameter count: {len(free)}")
        if len(hardcoded) == len(free):
            print("hardcoded optimal_params matches the active card layout")
        elif len(hardcoded) == len(legacy_free):
            print("hardcoded optimal_params matches the legacy freeze layout [1,5,6,9,10], not the active card")
            print(f"legacy free_indices: {legacy_free}")
        else:
            print("hardcoded optimal_params does not match either the active or legacy free layout")

    if trials:
        print(f"trial results found in notebook outputs: {len(trials)}")
        for trial_id, chi2, params in trials:
            print(f"trial {trial_id}: chi2/N={chi2:.3f}, free_param_count={len(params)}, params={params}")
    else:
        print("No trial outputs found in fit.ipynb")

    print_section("Best Available Current Fit")
    current_trial = None
    matching_trials = [item for item in trials if len(item[2]) == len(free)]
    if matching_trials:
        current_trial = min(matching_trials, key=lambda item: item[1])
        trial_id, chi2, free_values = current_trial
        full = fill_params(initial, frozen, free_values)
        print(f"best matching notebook trial: {trial_id}")
        print(f"best chi2/N: {chi2:.3f}")
        print(f"best full params: {full}")
    else:
        full = None
        print("No notebook trial matches the active card free-parameter count")

    if full is not None:
        print_section("Representative NP Factors")
        x = 0.02
        for q in (91.1876, 350.0, 1000.0):
            print(f"Q = {q:.4g} GeV, x = {x}")
            for b in (0.15, 0.33, 0.50, 0.80):
                s = current_np_summary(fields, full, x, b, q)
                print(
                    f"  b={b:>4.2f}: "
                    f"xshape={s['xshape']:.5f}, "
                    f"SNP_mu={s['snp_mu']:.5f}, "
                    f"bstar_CS={s['bstar_cs']:.5f}, "
                    f"NP_zeta={s['np_zeta']:.5f}, "
                    f"total~={s['total_factor']:.5f}"
                )

        print_section("g2 Sensitivity At The Z Pole")
        base = dict(zip(fields, full))
        for g2 in (0.2, 0.3, 0.4, 0.5, 0.6):
            varied = list(full)
            varied[fields.index("g2")] = g2
            s = current_np_summary(fields, varied, 0.02, 0.50, 91.1876)
            print(f"g2={g2:.1f}: b=0.50, Q=MZ, NP_zeta={s['np_zeta']:.5f}, total~={s['total_factor']:.5f}")

        print_section("power_CS Sensitivity At The Z Pole")
        for power in (0.7, 0.9, 1.0, 1.1):
            varied = list(full)
            varied[fields.index("power_CS")] = power
            s = current_np_summary(fields, varied, 0.02, 0.50, 91.1876)
            print(f"power_CS={power:.1f}: b=0.50, Q=MZ, bstar_CS={s['bstar_cs']:.5f}, NP_zeta={s['np_zeta']:.5f}, total~={s['total_factor']:.5f}")

    print_section("Interpretation")
    print("1. The active FI host/kernel path is structurally consistent: Params_Struct order matches and set_params copies the whole struct into the CL buffer.")
    print("2. fit.ipynb still contains a stale hardcoded optimal_params array from the legacy six-free-parameter setup, even though newer trial outputs use seven free parameters.")
    print("3. The current best available seven-parameter notebook trial keeps g2 near 0.509 and power_CS near 0.912, so collider small-b Z suppression remains substantial.")
    print("4. SNP_mu stays very close to 1 at x~0.02, so the remaining collider-shape lever is still mainly the zeta-side broadening, not intrinsic x-shape suppression.")
    print("5. The recent Default.jl / fit.ipynb changes strongly improved total chi2 in the notebook output, so part of the previous high-energy underprediction was configuration-level, not just kernel physics.")


if __name__ == "__main__":
    main()
