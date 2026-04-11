# Aggressive Table Refits

Heavier search than the earlier local/table refits:
- stage 1: many-start global search
- stage 2: top-4 global refinements
- stage 3: top-2 local refinements
- final polish

Tables scanned:
- `MSHT20N3LO-MC-0-2` with runtime `mustar` based on `n=0`
- `MSHT20N3LO-MC-4-2` with runtime `mustar` based on `n=4`
- `MSHT20N3LO-MC-freeze` with runtime `mustar` based on `n=0`

Best table per model:

- `poly_bstar_cslog`: best chi2 `MSHT20N3LO-MC-0-2` (0.834188), best absdev `MSHT20N3LO-MC-4-2` (0.051938), best shortfall `MSHT20N3LO-MC-4-2` (0.010810)
- `poly_bstar_cslog_loggauss`: best chi2 `MSHT20N3LO-MC-0-2` (0.821703), best absdev `MSHT20N3LO-MC-0-2` (0.051051), best shortfall `MSHT20N3LO-MC-4-2` (0.009500)
- `reduced_loggauss_powerseed`: best chi2 `MSHT20N3LO-MC-0-2` (0.825884), best absdev `MSHT20N3LO-MC-0-2` (0.051667), best shortfall `MSHT20N3LO-MC-0-2` (0.010491)
