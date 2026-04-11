from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FITS_DIR = SCRIPT_DIR.parent
ROOT = FITS_DIR.parent
CARDS_DIR = ROOT / "Cards"
NP_DIR = ROOT / "TMDs" / "NP Parameterizations"
RESULTS_ROOT = FITS_DIR / "results"
DEPRECATED_SCRIPTS_DIR = FITS_DIR / "deprecated" / "scripts"

for path in (SCRIPT_DIR, DEPRECATED_SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

RESULTS_ROOT.mkdir(exist_ok=True)
