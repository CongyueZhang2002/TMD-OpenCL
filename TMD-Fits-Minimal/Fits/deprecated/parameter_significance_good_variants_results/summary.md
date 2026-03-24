# Parameter Significance For Good Variants

Method:

- Local finite-difference Hessian around the fitted 0-2 point.
- Nested local refits with one parameter or one feature block turned off.
- `delta_chi2_total` is the main significance score for the nested tests.

Analyzed variants:

- `xb_bump_logb_nobump`: chi2/N `0.804696`, highE absdev `0.051043`, highE shortfall `0.010788`
- `xb_bump_logb_mult`: chi2/N `0.804752`, highE absdev `0.051334`, highE shortfall `0.010754`
- `reduced_loggauss_powerseed`: chi2/N `0.826231`, highE absdev `0.051434`, highE shortfall `0.010446`
- `entangle_window_exp`: chi2/N `0.833918`, highE absdev `0.050894`, highE shortfall `0.010852`

Scale parameters such as `logx0`, `sigx`, `logb0`, `sigb`, `bw`, and `BNP` are mainly interpretable through block tests.
See the per-model CSV files for top correlations and nested rankings.
