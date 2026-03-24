# Table Variant Scan

Scanned models:

- `baseline_unfrozen`: 0112 baseline
- `poly_bstar_cslog`: Poly-x bstar + CSlog
- `poly_bstar_cslog_loggauss`: Poly-x bstar + CSlog + loggauss

Tables scanned:

- `MSHT20N3LO-MC-0-0` with `n=0`, `m=0`
- `MSHT20N3LO-MC-0-2` with `n=0`, `m=2`
- `MSHT20N3LO-MC-4-2` with `n=4`, `m=2`
- `MSHT20N3LO-MC-4-4` with `n=4`, `m=4`

Per-model best table by metric:

- `baseline_unfrozen`: best `chi2` = `MSHT20N3LO-MC-0-2` (0.835061), best `absdev` = `MSHT20N3LO-MC-4-4` (0.054915), best `shortfall` = `MSHT20N3LO-MC-0-0` (0.018825)
- `poly_bstar_cslog`: best `chi2` = `MSHT20N3LO-MC-0-2` (0.835549), best `absdev` = `MSHT20N3LO-MC-4-2` (0.051464), best `shortfall` = `MSHT20N3LO-MC-4-4` (0.008760)
- `poly_bstar_cslog_loggauss`: best `chi2` = `MSHT20N3LO-MC-0-2` (0.822152), best `absdev` = `MSHT20N3LO-MC-0-2` (0.051222), best `shortfall` = `MSHT20N3LO-MC-4-4` (0.008579)

Aggregate mean ranks across the scanned models are in `aggregate_ranks.csv`.
