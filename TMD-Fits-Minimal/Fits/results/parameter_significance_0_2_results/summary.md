# Parameter Significance On Default 0-2

Method:

- Local finite-difference Hessian around the fitted 0-2 point.
- Nested local refits with one term turned off when there is a natural null/reference value.
- `delta_chi2_total` is the main contribution score for the nested tests.

Analyzed models:

- `poly_bstar_cslog_loggauss`: chi2/N `0.822117`, highE absdev `0.051243`, highE shortfall `0.010740`
- `baseline_unfrozen`: chi2/N `0.833860`, highE absdev `0.054914`, highE shortfall `0.020045`
- `poly_bstar_cslog`: chi2/N `0.834730`, highE absdev `0.052834`, highE shortfall `0.012694`

See per-model CSV files for parameter uncertainties, top correlations, and nested tests.
