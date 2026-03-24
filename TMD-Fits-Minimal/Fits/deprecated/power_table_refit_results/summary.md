# Power Table Refit Scan

Assumption: all power tables use the same runtime `mustar_func` choice as `MSHT20N3LO-MC-0-2`, i.e. `n=0`.

Scanned models:

- `baseline_unfrozen`: 0112 baseline
- `poly_bstar_cslog`: Poly-x bstar + CSlog
- `poly_bstar_cslog_loggauss`: Poly-x bstar + CSlog + loggauss

Tables scanned:

- `MSHT20N3LO-MC-0-2` (`baseline`), runtime `n=0`, reference `m=2`
- `MSHT20N3LO-MC-power-full` (`power_full`), runtime `n=0`, reference `m=2`
- `MSHT20N3LO-MC-power-x` (`power_x`), runtime `n=0`, reference `m=2`
- `MSHT20N3LO-MC-power-leptonic` (`power_leptonic`), runtime `n=0`, reference `m=2`

Best table per model:

- `baseline_unfrozen`: best chi2 `MSHT20N3LO-MC-0-2` (0.833860), best absdev `MSHT20N3LO-MC-0-2` (0.054914), best shortfall `MSHT20N3LO-MC-0-2` (0.020045)
- `poly_bstar_cslog`: best chi2 `MSHT20N3LO-MC-0-2` (0.834731), best absdev `MSHT20N3LO-MC-power-leptonic` (0.052823), best shortfall `MSHT20N3LO-MC-power-leptonic` (0.012445)
- `poly_bstar_cslog_loggauss`: best chi2 `MSHT20N3LO-MC-0-2` (0.822116), best absdev `MSHT20N3LO-MC-0-2` (0.051243), best shortfall `MSHT20N3LO-MC-power-leptonic` (0.010588)
