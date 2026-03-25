from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from julia import Main
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


FIT_NAME = "BroadBump42LogGaussAlpha1NoLambda2"
IF_GRID = True

MAP_PLOT_FLAVOR = "u"
MAP_X_VALUES = [1e-3, 1e-2, 1e-1]
MAP_Q_VALUES = [2.0, 10.0]

B_PLOT_MIN = 1e-3
B_PLOT_MAX = 8.0
N_B_PLOT = 180

KT_PLOT_MIN = 0.0
KT_PLOT_MAX = 2.0
N_KT_PLOT = 120

SURFACE_Q = 4.75
SURFACE_X_MIN = 1e-3
SURFACE_X_MAX = 0.99
N_SURFACE_X = 50
SURFACE_KT_MIN = 0.0
SURFACE_KT_MAX = 2.0
N_SURFACE_KT = 20
SURFACE_B_MIN = 0.05
SURFACE_B_MAX = 4.0
N_SURFACE_B = 40
SURFACE_Z_MIN = 0.0
SURFACE_Z_MAX = 1.0

OUT_DIR = Path(__file__).resolve().parent / "tmdpdf_plot_outputs" / FIT_NAME


def include(repo_root: Path, rel: str) -> None:
    path = (repo_root / rel).resolve()
    Main.eval(f'include(raw"{path}")')


def push_params(param_names: list[str], params: np.ndarray) -> None:
    lines = []
    for name, val in zip(param_names, params):
        lines.append(f"global NP_{name} = Float32({float(val)})")
    Main.eval("\n".join(lines))


def as_array(julia_tuple) -> np.ndarray:
    return np.asarray(julia_tuple, dtype=float)


def setup() -> tuple[Path, list[str], np.ndarray, int]:
    tmd_root = Path(__file__).resolve().parent
    repo_root = tmd_root.parent

    include(repo_root, f"Cards/{FIT_NAME}.jl")

    param_names = [str(x) for x in Main.eval("collect(fieldnames(Params_Struct))")]
    params = np.asarray(Main.initial_params, dtype=float)
    push_params(param_names, params)

    np_cl_name = str(Main.NP_name)
    np_julia_name = Path(np_cl_name).with_suffix(".jl").name
    np_julia_path = tmd_root / "NP Parameterizations Julia" / np_julia_name
    if not np_julia_path.exists():
        raise FileNotFoundError(f"Missing Julia NP file {np_julia_path}")

    Main.if_grid = bool(IF_GRID)
    try:
        Main.FastGK
    except Exception:
        Main.eval("module FastGK end")
    try:
        Main.eval("b0")
    except Exception:
        Main.eval("const b0 = 1.1229189")

    include(repo_root, "TMDs/Grids/initialization.jl")
    include(repo_root, f"TMDs/NP Parameterizations Julia/{np_julia_name}")
    push_params(param_names, params)
    include(repo_root, "TMDs/TMDPDFs/TMDPDFN.jl")

    flavors = ["u", "ub", "d", "db", "s", "sb", "c", "cb", "b", "bb"]
    flavor_index = {name: i for i, name in enumerate(flavors)}
    plot_flavor_idx = flavor_index[MAP_PLOT_FLAVOR]
    return repo_root, param_names, params, plot_flavor_idx


def eval_component_b(plot_flavor_idx: int, x_value: float, q_value: float, b_value: float) -> float:
    vals = as_array(Main.TMDPDF_func(b=float(b_value), x=float(x_value), Q=float(q_value)))
    return float(x_value * vals[plot_flavor_idx])


def eval_component_kt(plot_flavor_idx: int, x_value: float, q_value: float, kt_value: float) -> float:
    vals = as_array(Main.TMDPDF_kt_func(kt=float(kt_value), x=float(x_value), Q=float(q_value)))
    return float(x_value * vals[plot_flavor_idx])


def plot_slices(plot_flavor_idx: int) -> None:
    b_plot = np.linspace(B_PLOT_MIN, B_PLOT_MAX, N_B_PLOT)
    kt_plot = np.linspace(KT_PLOT_MIN, KT_PLOT_MAX, N_KT_PLOT)

    fig, axes = plt.subplots(
        nrows=len(MAP_Q_VALUES),
        ncols=2,
        figsize=(11.0, 4.2 * len(MAP_Q_VALUES)),
        constrained_layout=True,
    )
    if len(MAP_Q_VALUES) == 1:
        axes = np.asarray([axes])

    for row_idx, q_value in enumerate(MAP_Q_VALUES):
        ax_b = axes[row_idx, 0]
        ax_kt = axes[row_idx, 1]
        for x_value in MAP_X_VALUES:
            b_vals = np.array([eval_component_b(plot_flavor_idx, x_value, q_value, b) for b in b_plot])
            kt_vals = np.array([eval_component_kt(plot_flavor_idx, x_value, q_value, kt) for kt in kt_plot])
            label = f"x={x_value:g}"
            ax_b.plot(b_plot, b_vals, linewidth=1.8, label=label)
            ax_kt.plot(kt_plot, kt_vals, linewidth=1.8, label=label)

        ax_b.set_title(f"b-space, Q={q_value:g} GeV")
        ax_b.set_xlabel(r"$b\;[\mathrm{GeV}^{-1}]$")
        ax_b.set_ylabel(r"$x f_1^u(x,b;Q)$")
        ax_b.grid(True, alpha=0.3)

        ax_kt.set_title(f"kT-space, Q={q_value:g} GeV")
        ax_kt.set_xlabel(r"$k_T\;[\mathrm{GeV}]$")
        ax_kt.set_ylabel(r"$x f_1^u(x,k_T;Q)$")
        ax_kt.grid(True, alpha=0.3)
        ax_kt.legend(frameon=False, fontsize=9)

    fig.savefig(OUT_DIR / "map_slices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def configure_surface_axes(ax, y_label: str, y_min: float, y_max: float) -> None:
    x_ticks = np.log10(np.array([1e-3, 1e-2, 1e-1, 9e-1]))
    ax.set_xlim(np.log10(SURFACE_X_MIN), np.log10(SURFACE_X_MAX))
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(SURFACE_Z_MIN, SURFACE_Z_MAX)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0.001", "0.01", "0.1", "0.9"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(y_label)
    ax.set_zlabel(r"$x f_1^u$")
    ax.view_init(elev=28, azim=45)
    ax.set_box_aspect((1.4, 1.2, 1.0))


def plot_surfaces(plot_flavor_idx: int) -> None:
    surface_x = np.geomspace(SURFACE_X_MIN, SURFACE_X_MAX, N_SURFACE_X)
    surface_x_log = np.log10(surface_x)
    surface_b = np.linspace(SURFACE_B_MIN, SURFACE_B_MAX, N_SURFACE_B)
    surface_kt = np.linspace(SURFACE_KT_MIN, SURFACE_KT_MAX, N_SURFACE_KT)

    z_b = np.zeros((len(surface_b), len(surface_x)))
    z_kt = np.zeros((len(surface_kt), len(surface_x)))

    for ix, x_value in enumerate(surface_x):
        for ib, b_value in enumerate(surface_b):
            z_b[ib, ix] = eval_component_b(plot_flavor_idx, x_value, SURFACE_Q, b_value)
        for ik, kt_value in enumerate(surface_kt):
            z_kt[ik, ix] = eval_component_kt(plot_flavor_idx, x_value, SURFACE_Q, kt_value)

    z_b = np.clip(z_b, SURFACE_Z_MIN, SURFACE_Z_MAX)
    z_kt = np.clip(z_kt, SURFACE_Z_MIN, SURFACE_Z_MAX)

    xb, yb = np.meshgrid(surface_x_log, surface_b)
    xk, yk = np.meshgrid(surface_x_log, surface_kt)

    fig = plt.figure(figsize=(14.5, 6.4), constrained_layout=True)
    ax_b = fig.add_subplot(1, 2, 1, projection="3d")
    ax_kt = fig.add_subplot(1, 2, 2, projection="3d")

    surf_b = ax_b.plot_surface(xb, yb, z_b, cmap="Wistia", linewidth=0.45, edgecolor="0.55", antialiased=True)
    surf_kt = ax_kt.plot_surface(xk, yk, z_kt, cmap="Wistia", linewidth=0.45, edgecolor="0.55", antialiased=True)
    fig.colorbar(surf_b, ax=ax_b, shrink=0.78, pad=0.06)
    fig.colorbar(surf_kt, ax=ax_kt, shrink=0.78, pad=0.06)

    configure_surface_axes(ax_b, r"$b\;[\mathrm{GeV}^{-1}]$", SURFACE_B_MIN, SURFACE_B_MAX)
    configure_surface_axes(ax_kt, r"$k_T\;[\mathrm{GeV}]$", SURFACE_KT_MIN, SURFACE_KT_MAX)
    ax_b.set_title(f"3D b-space surface at Q={SURFACE_Q:g} GeV")
    ax_kt.set_title(f"3D kT-space surface at Q={SURFACE_Q:g} GeV")

    fig.savefig(OUT_DIR / "surfaces_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, param_names, params, plot_flavor_idx = setup()
    plot_slices(plot_flavor_idx)
    plot_surfaces(plot_flavor_idx)
    summary = {
        "fit_name": FIT_NAME,
        "param_names": param_names,
        "params": params.tolist(),
        "map_plot_flavor": MAP_PLOT_FLAVOR,
        "map_x_values": MAP_X_VALUES,
        "map_Q_values": MAP_Q_VALUES,
        "surface_Q": SURFACE_Q,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
