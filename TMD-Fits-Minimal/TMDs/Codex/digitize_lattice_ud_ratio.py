from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import fitz
import numpy as np


PDF_URL = "https://arxiv.org/pdf/2505.18430"
PAGE_INDEX = 3

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "_digitize_cache"
PDF_PATH = CACHE_DIR / "2505.18430.pdf"
CSV_PATH = ROOT / "lattice_ud_ratio_fig4.csv"


SERIES = {
    0.2: {
        "color": (0.8627451062202454, 0.0784313753247261, 0.23529411852359772),
        "marker": "square",
    },
    0.3: {
        "color": (0.2549019455909729, 0.4117647111415863, 0.8823529481887817),
        "marker": "circle",
    },
    0.4: {
        "color": (0.0, 0.5019607543945312, 0.0),
        "marker": "diamond",
    },
}


def _close_color(color: tuple[float, float, float] | None, target: tuple[float, float, float], tol: float = 1e-4) -> bool:
    if color is None:
        return False
    return all(abs(a - b) <= tol for a, b in zip(color, target))


def ensure_pdf() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    return PDF_PATH


def linear_map(px_values: list[float], data_values: list[float]) -> np.ndarray:
    return np.polyfit(np.asarray(px_values, dtype=float), np.asarray(data_values, dtype=float), 1)


def apply_map(coeffs: np.ndarray, value: float) -> float:
    return float(np.polyval(coeffs, value))


def main() -> None:
    pdf_path = ensure_pdf()
    pdf = fitz.open(pdf_path)
    page = pdf[PAGE_INDEX]
    drawings = page.get_drawings()

    plot_rect = fitz.Rect(341.23, 466.73, 548.99, 600.18)

    x_tick_px = [363.10, 406.84, 450.58, 494.32, 538.06]
    x_tick_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    y_tick_px = [592.56, 554.43, 516.30, 478.17]
    y_tick_values = [1.0, 2.0, 3.0, 4.0]

    x_map = linear_map(x_tick_px, x_tick_values)
    y_map = linear_map(y_tick_px, y_tick_values)

    rows: list[dict[str, float | str]] = []

    for x_value, meta in SERIES.items():
        color = meta["color"]

        error_bars = [
            d
            for d in drawings
            if d.get("type") == "s"
            and _close_color(d.get("color"), color)
            and abs(d["rect"].x1 - d["rect"].x0) < 1e-6
            and d["rect"].height > 5.0
            and d["rect"].x0 >= plot_rect.x0
            and d["rect"].x0 <= plot_rect.x1
            and d["rect"].y0 >= 530.0
            and d["rect"].y1 <= 590.0
        ]

        markers = [
            d
            for d in drawings
            if d.get("type") == "fs"
            and _close_color(d.get("fill"), color)
            and d["rect"].width > 3.0
            and d["rect"].height > 3.0
            and d["rect"].width < 8.0
            and d["rect"].height < 8.0
            and d["rect"].x0 >= plot_rect.x0
            and d["rect"].x1 <= plot_rect.x1
            and d["rect"].y0 >= 530.0
            and d["rect"].y1 <= 580.0
        ]

        error_bars.sort(key=lambda d: d["rect"].x0)
        markers.sort(key=lambda d: 0.5 * (d["rect"].x0 + d["rect"].x1))

        if len(error_bars) != len(markers):
            raise RuntimeError(
                f"Series x={x_value:g}: found {len(error_bars)} error bars and {len(markers)} markers"
            )

        for error_bar, marker in zip(error_bars, markers):
            x_px = float(error_bar["rect"].x0)
            marker_center_y = 0.5 * (marker["rect"].y0 + marker["rect"].y1)
            b_fm = apply_map(x_map, x_px)
            central = apply_map(y_map, marker_center_y)
            std = 0.5 * abs(apply_map(y_map, error_bar["rect"].y0) - apply_map(y_map, error_bar["rect"].y1))

            rows.append(
                {
                    "x": round(x_value, 1),
                    "marker": meta["marker"],
                    "b_fm": b_fm,
                    "central": central,
                    "std": std,
                }
            )

    rows.sort(key=lambda row: (row["x"], row["b_fm"]))

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "marker", "b_fm", "central", "std"])
        writer.writeheader()
        writer.writerows(rows)

    print(CSV_PATH)


if __name__ == "__main__":
    main()
