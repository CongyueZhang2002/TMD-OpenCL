# Freeze Table Refit Results

Table: `MSHT20N3LO-MC-freeze`

Assumption used in the generated kernels:

- `mustar_func(b, Q) = max(bmax / b, 1.0f)`, matching the latest requested freeze-table check.

Models scanned:

- `baseline_unfrozen`: 0112 baseline
- `poly_bstar_cslog`: Poly-x bstar + CSlog
- `reduced_loggauss_powerseed`: Reduced loggauss powerseed
- `xb_bump_logb_nobump`: XB logb no-bump

Results:

- `poly_bstar_cslog`: freeze chi2/N = `1.075633`, absdev = `0.052855`, shortfall = `0.011501`, delta vs 0-2 chi2 = `+0.240048`, delta vs 0-2 absdev = `+0.000089`, delta vs 0-2 shortfall = `-0.000998`
- `reduced_loggauss_powerseed`: freeze chi2/N = `1.096692`, absdev = `0.053597`, shortfall = `0.010486`, delta vs 0-2 chi2 = `+0.270460`, delta vs 0-2 absdev = `+0.002162`, delta vs 0-2 shortfall = `+0.000039`
- `xb_bump_logb_nobump`: freeze chi2/N = `1.081215`, absdev = `0.054321`, shortfall = `0.010782`, delta vs 0-2 chi2 = `+0.276519`, delta vs 0-2 absdev = `+0.003278`, delta vs 0-2 shortfall = `-0.000006`
- `baseline_unfrozen`: freeze chi2/N = `1.020360`, absdev = `0.057241`, shortfall = `0.022448`, delta vs 0-2 chi2 = `+0.185182`, delta vs 0-2 absdev = `+0.002222`, delta vs 0-2 shortfall = `+0.002323`
