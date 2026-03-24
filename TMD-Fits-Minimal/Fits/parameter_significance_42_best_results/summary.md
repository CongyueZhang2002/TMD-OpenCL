# Parameter Significance For 4-2 Best No-Bump Model

- Model: `Poly-x bstar + CSlog (4-2 best, no bump)`
- Fit name: `Art23FamilyMuPolyBstarCSLog42Best`
- chi2/N: `1.029358`
- total chi2: `478.651`
- highE absdev: `0.051938`
- highE shortfall: `0.010810`

Method:

- Local finite-difference Hessian around the saved aggressive-refit 4-2 point.
- PSD-projected covariance from the chi2 Hessian.
- Nested local refits with one term set to its natural null/reference value.

Saved files:

- `poly_bstar_cslog_42best_parameter_uncertainty.csv`
- `poly_bstar_cslog_42best_correlation_matrix.csv`
- `poly_bstar_cslog_42best_top_correlations.csv`
- `poly_bstar_cslog_42best_hessian_total.csv`
- `poly_bstar_cslog_42best_nested_tests.csv`
