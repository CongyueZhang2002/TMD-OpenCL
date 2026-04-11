#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SOURCE_NOTEBOOK_URL = (
    "https://github.com/CZhou-24/TMD-CUDA-main/blob/main/"
    "TMD-Fits-Minimal/Fits/fit_bayesian.ipynb"
)
SOURCE_CARD_URL = (
    "https://github.com/CZhou-24/TMD-CUDA-main/blob/main/"
    "TMD-Fits-Minimal/Cards/BroadBump42LogGaussAlpha1NoLambda2.jl"
)
DEFAULT_BURN = 1000
DEFAULT_THIN = 1


@dataclass(frozen=True)
class ParameterSpec:
    parameterization: str
    chain_parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    chain_storage: str = "normalized"
    source_note: str = ""

    @property
    def ndim(self) -> int:
        return len(self.chain_parameter_names)

    def denormalize(self, samples: np.ndarray) -> np.ndarray:
        if self.chain_storage == "physical":
            return samples
        if self.chain_storage != "normalized":
            raise ValueError(
                f"Unsupported chain_storage={self.chain_storage!r}; "
                "use 'normalized' or 'physical'."
            )

        lower = np.asarray(self.lower_bounds, dtype=np.float64)
        upper = np.asarray(self.upper_bounds, dtype=np.float64)
        if samples.shape[1] != lower.shape[0]:
            raise ValueError(
                "Parameter/bounds mismatch while denormalizing samples: "
                f"samples ndim={samples.shape[1]}, spec ndim={lower.shape[0]}."
            )
        return lower[None, :] + samples * (upper - lower)[None, :]


KNOWN_PARAMETERIZATIONS: dict[str, ParameterSpec] = {
    "broadbump42loggaussalpha1nolambda2": ParameterSpec(
        parameterization="BroadBump42LogGaussAlpha1NoLambda2",
        chain_parameter_names=(
            "lambda1",
            "lambda2",
            "lambda3",
            "logx0",
            "sigx",
            "amp",
            "BNP",
            "c0",
            "c1",
        ),
        lower_bounds=(-0.5, 0.02, -10.0, -9.210340372, 0.6, -3.0, 0.4, 0.0, 0.0),
        upper_bounds=(0.5, 8.0, 10.0, -1.203972804, 2.5, 3.0, 4.5, 0.25, 0.25),
        chain_storage="normalized",
        source_note=(
            "Verified from the source fit_bayesian.ipynb and "
            "Cards/BroadBump42LogGaussAlpha1NoLambda2.jl."
        ),
    ),
    "bb42loggaussalpha1nolambda2": ParameterSpec(
        parameterization="BroadBump42LogGaussAlpha1NoLambda2",
        chain_parameter_names=(
            "lambda1",
            "lambda2",
            "lambda3",
            "logx0",
            "sigx",
            "amp",
            "BNP",
            "c0",
            "c1",
        ),
        lower_bounds=(-0.5, 0.02, -10.0, -9.210340372, 0.6, -3.0, 0.4, 0.0, 0.0),
        upper_bounds=(0.5, 8.0, 10.0, -1.203972804, 2.5, 3.0, 4.5, 0.25, 0.25),
        chain_storage="normalized",
        source_note=(
            "Alias for BroadBump42LogGaussAlpha1NoLambda2; verified from the "
            "source notebook/card."
        ),
    ),
}


def normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract flat named Bayesian parameter samples from emcee HDF5 "
            "backends. For the inspected source repo, the verified built-in "
            "mapping is the 9-parameter BroadBump42LogGaussAlpha1NoLambda2 "
            "model. Ambiguous chains such as the source 6d files require an "
            "explicit JSON parameter spec."
        )
    )
    parser.add_argument(
        "chains",
        nargs="*",
        help="One or more emcee HDF5 backend files to extract.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for generated CSV files. Default: this script's folder.",
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=DEFAULT_BURN,
        help=f"Discard this many initial steps. Default: {DEFAULT_BURN}.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=DEFAULT_THIN,
        help=f"Retain every Nth step after burn-in. Default: {DEFAULT_THIN}.",
    )
    parser.add_argument(
        "--parameterization",
        help=(
            "Known parameterization name or alias. If omitted, 9d chains "
            "default to BroadBump42LogGaussAlpha1NoLambda2."
        ),
    )
    parser.add_argument(
        "--parameter-spec",
        help=(
            "Path to a JSON file describing the chain parameter order and bounds. "
            "Use this for ambiguous chains such as the source 6d files."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional explicit CSV path. Only valid when extracting a single chain.",
    )
    parser.add_argument(
        "--list-known-parameterizations",
        action="store_true",
        help="List built-in verified parameterizations and exit.",
    )
    return parser.parse_args()


def print_known_parameterizations() -> None:
    seen: set[str] = set()
    for spec in KNOWN_PARAMETERIZATIONS.values():
        if spec.parameterization in seen:
            continue
        seen.add(spec.parameterization)
        print(f"{spec.parameterization} ({spec.ndim}d)")
        print(f"  names: {', '.join(spec.chain_parameter_names)}")
        print(f"  source: {spec.source_note}")


def load_custom_parameter_spec(path: Path) -> ParameterSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Parameter spec not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON parameter spec {path}: {exc}") from exc

    parameterization = str(payload.get("parameterization", "CustomParameterization"))
    chain_storage = str(payload.get("chain_storage", "normalized"))
    source_note = str(payload.get("source_note", ""))

    if "parameters" in payload:
        params = payload["parameters"]
        if not isinstance(params, list) or not params:
            raise ValueError(
                f"{path} must define a non-empty 'parameters' list when that key is used."
            )
        names = []
        lower = []
        upper = []
        for item in params:
            if not isinstance(item, dict):
                raise ValueError(f"{path} has a non-object entry in 'parameters': {item!r}")
            if "name" not in item or "lower" not in item or "upper" not in item:
                raise ValueError(
                    f"{path} entries in 'parameters' must include name/lower/upper."
                )
            names.append(str(item["name"]))
            lower.append(float(item["lower"]))
            upper.append(float(item["upper"]))
    else:
        names = [str(x) for x in payload.get("chain_parameter_names", [])]
        lower = [float(x) for x in payload.get("lower_bounds", [])]
        upper = [float(x) for x in payload.get("upper_bounds", [])]

    if not names:
        raise ValueError(
            f"{path} does not define any chain parameter names. "
            "Use either 'parameters' or 'chain_parameter_names'."
        )
    if len(names) != len(lower) or len(names) != len(upper):
        raise ValueError(
            f"{path} must provide the same number of parameter names, lower bounds, "
            f"and upper bounds; got {len(names)}, {len(lower)}, {len(upper)}."
        )

    return ParameterSpec(
        parameterization=parameterization,
        chain_parameter_names=tuple(names),
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
        chain_storage=chain_storage,
        source_note=source_note,
    )


def resolve_parameter_spec(
    ndim: int,
    parameterization_name: str | None,
    parameter_spec_path: str | None,
) -> ParameterSpec:
    if parameter_spec_path:
        spec = load_custom_parameter_spec(Path(parameter_spec_path))
    elif parameterization_name:
        key = normalize_key(parameterization_name)
        if key not in KNOWN_PARAMETERIZATIONS:
            known = ", ".join(sorted({spec.parameterization for spec in KNOWN_PARAMETERIZATIONS.values()}))
            raise ValueError(
                f"Unknown parameterization {parameterization_name!r}. "
                f"Known built-ins: {known}. "
                "For custom or ambiguous chains, pass --parameter-spec."
            )
        spec = KNOWN_PARAMETERIZATIONS[key]
    elif ndim == 9:
        spec = KNOWN_PARAMETERIZATIONS["broadbump42loggaussalpha1nolambda2"]
    else:
        raise ValueError(
            "Could not infer a verified parameter mapping for this chain. "
            f"The chain has ndim={ndim}. The inspected source notebook/card only "
            "verify the 9d BroadBump42LogGaussAlpha1NoLambda2 map. "
            "Use --parameter-spec for ambiguous chains such as the source 6d files."
        )

    if spec.ndim != ndim:
        raise ValueError(
            f"Chain dimensionality mismatch: chain ndim={ndim}, "
            f"parameter spec ndim={spec.ndim} for {spec.parameterization!r}."
        )
    return spec


def read_emcee_backend(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    with h5py.File(path, "r") as handle:
        if "mcmc" not in handle:
            raise ValueError(f"{path} is missing the 'mcmc' group expected from emcee.")
        group = handle["mcmc"]
        if "chain" not in group:
            raise ValueError(f"{path} is missing mcmc/chain.")

        chain = np.asarray(group["chain"], dtype=np.float64)
        if chain.ndim != 3:
            raise ValueError(
                f"{path} has unexpected mcmc/chain shape {chain.shape}; "
                "expected (nsteps, nwalkers, ndim)."
            )

        log_prob: np.ndarray | None = None
        if "log_prob" in group:
            log_prob = np.asarray(group["log_prob"], dtype=np.float64)
        else:
            print(
                f"[note] {path.name}: mcmc/log_prob is unavailable; omitting log_prob column.",
                file=sys.stderr,
            )

        if "blobs" not in group:
            print(
                f"[note] {path.name}: mcmc/blobs is unavailable; "
                "the inspected source chains did not rely on blobs.",
                file=sys.stderr,
            )

    return chain, log_prob


def flatten_chain(
    chain: np.ndarray,
    log_prob: np.ndarray | None,
    burn: int,
    thin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    nsteps, nwalkers, ndim = chain.shape
    if burn < 0:
        raise ValueError(f"Burn-in must be non-negative, got {burn}.")
    if thin <= 0:
        raise ValueError(f"Thin must be positive, got {thin}.")
    if burn >= nsteps:
        raise ValueError(
            f"No samples remain after burn-in: burn={burn}, nsteps={nsteps}. "
            "Lower --burn if you intentionally want to inspect this backend."
        )

    kept_chain = chain[burn::thin]
    if kept_chain.shape[0] == 0:
        raise ValueError(
            f"No samples remain after applying burn={burn} and thin={thin}."
        )

    kept_steps = np.arange(burn, nsteps, thin, dtype=np.int64)
    flat_chain = kept_chain.reshape(-1, ndim)
    walker = np.tile(np.arange(nwalkers, dtype=np.int64), kept_chain.shape[0])
    step = np.repeat(kept_steps, nwalkers)

    flat_log_prob: np.ndarray | None = None
    if log_prob is not None:
        if log_prob.shape != chain.shape[:2]:
            raise ValueError(
                f"log_prob shape mismatch: chain has {chain.shape[:2]}, "
                f"log_prob has {log_prob.shape}."
            )
        flat_log_prob = log_prob[burn::thin].reshape(-1)

    return flat_chain, walker, step, flat_log_prob


def make_output_path(output_dir: Path, chain_path: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    return output_dir / f"bayesian_parameters_{chain_path.stem}.csv"


def extract_chain(
    chain_path: Path,
    output_dir: Path,
    burn: int,
    thin: int,
    parameterization_name: str | None,
    parameter_spec_path: str | None,
    explicit_output: str | None,
) -> Path:
    chain, log_prob = read_emcee_backend(chain_path)
    ndim = int(chain.shape[-1])
    spec = resolve_parameter_spec(ndim, parameterization_name, parameter_spec_path)
    flat_chain, walker, step, flat_log_prob = flatten_chain(chain, log_prob, burn, thin)
    physical = spec.denormalize(flat_chain)

    data: dict[str, Any] = {
        "sample_id": np.arange(flat_chain.shape[0], dtype=np.int64),
        "walker": walker,
        "step": step,
    }
    if flat_log_prob is not None:
        data["log_prob"] = flat_log_prob

    for idx, name in enumerate(spec.chain_parameter_names):
        data[name] = physical[:, idx]

    df = pd.DataFrame(data)
    output_path = make_output_path(output_dir, chain_path, explicit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Wrote {output_path} "
        f"({df.shape[0]} rows, {df.shape[1]} columns) from {chain_path.name} "
        f"using {spec.parameterization}."
    )
    if flat_log_prob is None:
        print(f"[note] {chain_path.name}: output omits log_prob because it was unavailable.")
    return output_path


def main() -> int:
    args = parse_args()

    if args.list_known_parameterizations:
        print_known_parameterizations()
        return 0

    if not args.chains:
        print("No chain files were provided. Pass one or more HDF5 paths.", file=sys.stderr)
        return 2

    if args.output and len(args.chains) != 1:
        print("--output can only be used with a single input chain.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()

    try:
        for raw_chain in args.chains:
            extract_chain(
                chain_path=Path(raw_chain).resolve(),
                output_dir=output_dir,
                burn=args.burn,
                thin=args.thin,
                parameterization_name=args.parameterization,
                parameter_spec_path=args.parameter_spec,
                explicit_output=args.output,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
