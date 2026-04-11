from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auto_np_search import CARDS_DIR, FITS_DIR, HIGH_ENERGY_FILES, CandidateSpec, FitSession
from parameter_significance_0_2 import finite_diff_hessian, psd_covariance, top_correlations
from parameter_significance_good_variants import MODELS as GOOD_MODELS, make_spec as make_good_spec
from scan_table_variants import parse_array, parse_struct_fields
from _paths import RESULTS_ROOT, ROOT


NP_DIR = ROOT / "TMDs" / "NP Parameterizations"
PLOT_DIR = ROOT / "Plot"
RESULTS_DIR = RESULTS_ROOT / "powerseed_hessian_analysis_results"
RESULTS_DIR.mkdir(exist_ok=True)

SOURCE_CARD = CARDS_DIR / "ReducedLogGaussPowerSeed.jl"
SOURCE_KERNEL = NP_DIR / "NP-ReducedLogGaussPowerSeed.cl"
SOURCE_RESULT = RESULTS_ROOT / "bump_variant_followup_results" / "reduced_loggauss_powerseed.json"

ANALYSIS_FIT_NAME = "PowerseedHessianAnalysis"
ANALYSIS_CARD = CARDS_DIR / f"{ANALYSIS_FIT_NAME}.jl"
ANALYSIS_KERNEL = NP_DIR / f"NP-{ANALYSIS_FIT_NAME}.cl"
DEFAULT_REL_STEP = 5e-4


def load_best_full() -> list[float]:
    data = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    block = data.get("best") or data["fit"]
    if "full_params" in block:
        return [float(x) for x in block["full_params"]]
    return [float(x) for x in block["free_params"]]


def replace_bracket_assignment(text: str, name: str, values: list[float]) -> str:
    import re

    rendered = ", ".join(f"{v:.12g}" for v in values)
    pattern = rf"(?ms)^([ \t]*(?!#){re.escape(name)}\s*=\s*)\[(.*?)\]"
    return re.sub(pattern, rf"\1[{rendered}]", text, count=1)


def replace_scalar_assignment(text: str, name: str, value: str) -> str:
    import re

    pattern = rf'(?m)^([ \t]*(?!#){re.escape(name)}\s*=\s*)".*?"'
    return re.sub(pattern, rf'\1"{value}"', text, count=1)


def prepare_spec() -> CandidateSpec:
    card_text = SOURCE_CARD.read_text(encoding="utf-8")
    kernel_text = SOURCE_KERNEL.read_text(encoding="utf-8")
    param_names = parse_struct_fields(card_text)
    best_full = load_best_full()
    bounds = [(float(lo), float(hi)) for lo, hi in parse_array(card_text, "bounds_raw")]
    frozen = [int(x) for x in parse_array(card_text, "frozen_indices")]

    card_text = replace_scalar_assignment(card_text, "const NP_name", ANALYSIS_KERNEL.name)
    card_text = replace_bracket_assignment(card_text, "initial_params", best_full)
    ANALYSIS_CARD.write_text(card_text, encoding="utf-8")
    ANALYSIS_KERNEL.write_text(kernel_text, encoding="utf-8")

    return CandidateSpec(
        name="powerseed_hessian",
        fit_name=ANALYSIS_FIT_NAME,
        np_name=ANALYSIS_KERNEL.name,
        param_names=param_names,
        initial_params=best_full,
        bounds=bounds,
        frozen_indices=frozen,
        kernel_variant="powerseed_hessian",
    )


def build_core(rel_step: float = DEFAULT_REL_STEP):
    model = next(m for m in GOOD_MODELS if m.name == "reduced_loggauss_powerseed")
    spec = make_good_spec(model)
    session = FitSession(spec)
    best_eval = session.evaluate_free(session.initial_params[session.free_idx])
    n_total = int(sum(session.n_list.values()))

    h = finite_diff_hessian(session.objective_log_normalized, session.theta0, rel_step=rel_step)
    chi2dN = float(best_eval["metrics"]["chi2dN_total"])
    chi2_total = chi2dN * n_total

    ln10 = np.log(10.0)
    H_dN = chi2dN * (ln10 * h["H_log"] + (ln10**2) * np.outer(h["grad"], h["grad"]))
    H_total = n_total * H_dN
    H_psd, cov_norm = psd_covariance(H_total)

    scales = session.upper_bounds - session.lower_bounds
    cov_phys = np.diag(scales) @ cov_norm @ np.diag(scales)
    sigma_phys = np.sqrt(np.maximum(np.diag(cov_phys), 0.0))
    corr = cov_norm / np.sqrt(np.outer(np.maximum(np.diag(cov_norm), 1e-300), np.maximum(np.diag(cov_norm), 1e-300)))

    eig_h, vec_h = np.linalg.eigh(H_psd)
    order_h = np.argsort(eig_h)  # softest -> stiffest
    eig_h = eig_h[order_h]
    vec_h = vec_h[:, order_h]

    eig_c, vec_c = np.linalg.eigh(cov_norm)
    order_c = np.argsort(eig_c)[::-1]  # softest -> stiffest
    eig_c = eig_c[order_c]
    vec_c = vec_c[:, order_c]

    mode_shifts_norm = vec_c * np.sqrt(np.maximum(eig_c, 0.0))
    mode_shifts_phys = scales[:, None] * mode_shifts_norm
    mode_loading_over_sigma = mode_shifts_phys / np.maximum(sigma_phys[:, None], 1e-300)

    return {
        "spec": spec,
        "session": session,
        "best_eval": best_eval,
        "n_total": n_total,
        "chi2_total": chi2_total,
        "hessian": H_total,
        "cov_norm": cov_norm,
        "cov_phys": cov_phys,
        "corr": corr,
        "sigma_phys": sigma_phys,
        "eig_h": eig_h,
        "vec_h": vec_h,
        "eig_c": eig_c,
        "vec_c": vec_c,
        "mode_shifts_norm": mode_shifts_norm,
        "mode_shifts_phys": mode_shifts_phys,
        "mode_loading_over_sigma": mode_loading_over_sigma,
        "hessian_raw": h,
        "rel_step": rel_step,
    }


def save_tables(core: dict) -> None:
    session = core["session"]
    best_eval = core["best_eval"]
    param_names = session.free_param_names

    summary = {
        "model": "reduced_loggauss_powerseed",
        "fit_name": core["spec"].fit_name,
        "n_total": core["n_total"],
        "chi2dN_total": best_eval["metrics"]["chi2dN_total"],
        "chi2_total": core["chi2_total"],
        "highE_mean_absdev_first3": best_eval["metrics"]["highE_mean_absdev_first3"],
        "highE_mean_shortfall_first3": best_eval["metrics"]["highE_mean_shortfall_first3"],
        "rel_step": core["rel_step"],
        "hessian_evals": core["hessian_raw"]["nevals"],
        "grad_norm_log10": float(np.linalg.norm(core["hessian_raw"]["grad"])),
        "hessian_min_eig": float(np.min(core["eig_h"])),
        "hessian_max_eig": float(np.max(core["eig_h"])),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sigma_norm = np.sqrt(np.maximum(np.diag(core["cov_norm"]), 0.0))
    param_rows = []
    for name, value, sigma_n, sigma_p in zip(param_names, best_eval["free_params"], sigma_norm, core["sigma_phys"]):
        param_rows.append(
            {
                "param": name,
                "value": float(value),
                "sigma_norm": float(sigma_n),
                "sigma_phys": float(sigma_p),
                "frac_uncertainty": float(sigma_p / abs(value)) if abs(value) > 1e-12 else np.nan,
            }
        )
    pd.DataFrame(param_rows).to_csv(RESULTS_DIR / "parameter_summary.csv", index=False)

    pd.DataFrame(core["corr"], index=param_names, columns=param_names).to_csv(RESULTS_DIR / "correlation_matrix.csv")
    pd.DataFrame(core["cov_phys"], index=param_names, columns=param_names).to_csv(RESULTS_DIR / "covariance_physical.csv")
    pd.DataFrame(core["hessian"], index=param_names, columns=param_names).to_csv(RESULTS_DIR / "hessian_total.csv")
    pd.DataFrame(top_correlations(core["corr"], param_names, top_n=18)).to_csv(
        RESULTS_DIR / "top_correlations.csv", index=False
    )

    mode_rows = []
    for j in range(len(param_names)):
        row = {
            "mode": j + 1,
            "curvature_eig": float(core["eig_h"][j]),
            "variance_eig_norm": float(core["eig_c"][j]),
        }
        for i, name in enumerate(param_names):
            row[f"{name}_shift_phys"] = float(core["mode_shifts_phys"][i, j])
            row[f"{name}_loading_over_sigma"] = float(core["mode_loading_over_sigma"][i, j])
        mode_rows.append(row)
    pd.DataFrame(mode_rows).to_csv(RESULTS_DIR / "mode_summary.csv", index=False)


def plot_correlation_matrix(core: dict) -> None:
    param_names = core["session"].free_param_names
    corr = core["corr"]

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdYlBu_r")
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)
    ax.set_title("Powerseed Correlation Matrix")

    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hessian_spectrum(core: dict) -> None:
    eig_h = core["eig_h"]
    eig_c = core["eig_c"]
    modes = np.arange(1, len(eig_h) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].semilogy(modes, eig_h, marker="o", color="tab:red")
    axes[0].set_xlabel("Mode (Softest to Stiffest)")
    axes[0].set_ylabel("Hessian Eigenvalue")
    axes[0].set_title("Hessian Curvatures")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(modes, eig_c, marker="o", color="tab:blue")
    axes[1].set_xlabel("Mode (Softest to Stiffest)")
    axes[1].set_ylabel("Covariance Eigenvalue")
    axes[1].set_title("Mode Variances")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hessian_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mode_loadings(core: dict) -> None:
    param_names = core["session"].free_param_names
    load = core["mode_loading_over_sigma"]

    vmax = float(np.max(np.abs(load)))
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    im = ax.imshow(load.T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticks(range(load.shape[1]))
    ax.set_yticklabels([f"Mode {i+1}" for i in range(load.shape[1])])
    ax.set_title("1-sigma Hessian Mode Loadings / Marginal sigma")

    for i in range(load.shape[1]):
        for j in range(load.shape[0]):
            ax.text(j, i, f"{load[j, i]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signed loading / sigma(param)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mode_loadings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _predict_ratios(session: FitSession, full_params: np.ndarray, file_names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    predictions, _ = session._predict(full_params)
    out: dict[str, dict[str, np.ndarray]] = {}
    for file_name in file_names:
        df = session.data_list[file_name].reset_index(drop=True)
        data = df["xsec"].to_numpy(dtype=float)
        qT = df["qT_mean"].to_numpy(dtype=float)
        pred = np.asarray(predictions[file_name], dtype=float)
        out[file_name] = {
            "qT": qT,
            "ratio": pred / data,
        }
    return out


def save_mode_variations(core: dict, n_modes: int = 4) -> None:
    session = core["session"]
    theta0 = session.theta0.copy()
    baseline_full = np.asarray(core["best_eval"]["full_params"], dtype=float)
    baseline = _predict_ratios(session, baseline_full, HIGH_ENERGY_FILES)

    rows = []
    soft_modes = min(n_modes, len(core["eig_c"]))
    for mode_idx in range(soft_modes):
        dtheta = core["mode_shifts_norm"][:, mode_idx]
        for sign in (-1.0, 1.0):
            theta = np.clip(theta0 + sign * dtheta, 0.0, 1.0)
            free_params = session.denormalize_params(theta)
            full_params = session.full_from_free(free_params)
            pred = _predict_ratios(session, full_params, HIGH_ENERGY_FILES)
            for file_name in HIGH_ENERGY_FILES:
                qT = pred[file_name]["qT"]
                ratio = pred[file_name]["ratio"]
                for i in range(len(qT)):
                    rows.append(
                        {
                            "mode": mode_idx + 1,
                            "sign": "+" if sign > 0 else "-",
                            "file": file_name,
                            "bin_index": i,
                            "qT": float(qT[i]),
                            "ratio": float(ratio[i]),
                        }
                    )

        for file_name in HIGH_ENERGY_FILES:
            qT = baseline[file_name]["qT"]
            ratio = baseline[file_name]["ratio"]
            for i in range(len(qT)):
                rows.append(
                    {
                        "mode": mode_idx + 1,
                        "sign": "0",
                        "file": file_name,
                        "bin_index": i,
                        "qT": float(qT[i]),
                        "ratio": float(ratio[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "soft_mode_variations_highE.csv", index=False)

    nrows = soft_modes
    ncols = len(HIGH_ENERGY_FILES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.7 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for r in range(nrows):
        for c, file_name in enumerate(HIGH_ENERGY_FILES):
            ax = axes[r, c]
            block = df[(df["mode"] == r + 1) & (df["file"] == file_name)]
            block0 = block[block["sign"] == "0"].sort_values("bin_index")
            blockp = block[block["sign"] == "+"].sort_values("bin_index")
            blockm = block[block["sign"] == "-"].sort_values("bin_index")

            qT = block0["qT"].to_numpy(dtype=float)
            y0 = block0["ratio"].to_numpy(dtype=float)
            yp = blockp["ratio"].to_numpy(dtype=float)
            ym = blockm["ratio"].to_numpy(dtype=float)

            ax.fill_between(qT, np.minimum(ym, yp), np.maximum(ym, yp), color="tab:orange", alpha=0.28)
            ax.plot(qT, y0, color="black", linewidth=1.4)
            ax.plot(qT, yp, color="tab:red", linestyle="--", linewidth=1.1)
            ax.plot(qT, ym, color="tab:blue", linestyle="--", linewidth=1.1)
            ax.axhline(1.0, color="0.4", linewidth=0.9)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(Path(file_name).stem)
            if c == 0:
                ax.set_ylabel(f"Mode {r+1}\npred/data")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("qT")

    fig.suptitle("Powerseed Hessian Soft-Mode Variations", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RESULTS_DIR / "soft_mode_variations_highE.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(rel_step: float = DEFAULT_REL_STEP) -> dict:
    core = build_core(rel_step=rel_step)
    save_tables(core)
    plot_correlation_matrix(core)
    plot_hessian_spectrum(core)
    plot_mode_loadings(core)
    save_mode_variations(core, n_modes=4)
    return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel-step", type=float, default=DEFAULT_REL_STEP)
    parser.add_argument("--hard-exit", action="store_true")
    args = parser.parse_args()
    run_analysis(rel_step=args.rel_step)
    print(f"Saved results to {RESULTS_DIR}")
    sys.stdout.flush()
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
