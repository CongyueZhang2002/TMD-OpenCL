# Parameterization Record (2026-03-22)

This file records the parameterizations tested in the FI search and their saved results before cleanup.

- Plain fit table: [parameterization_record_2026-03-22_plain.csv](c:/Users/congyue zhang/Desktop/OpenCL fitter/TMD-Fits-Minimal/Fits/parameterization_record_2026-03-22_plain.csv)
- Weighted/Pareto table: [parameterization_record_2026-03-22_weighted.csv](c:/Users/congyue zhang/Desktop/OpenCL fitter/TMD-Fits-Minimal/Fits/parameterization_record_2026-03-22_weighted.csv)
- Plain runs recorded: 50
- Weighted runs recorded: 60

## Metrics

- `chi2dN_total`: global fit objective used in the plain searches
- `highE_mean_absdev_first3`: mean absolute `pred/data - 1` over the first 3 bins of the 5 high-energy datasets
- `highE_mean_shortfall_first3`: mean underprediction-only penalty over the same 15 bins

## Best Plain Fits By Total Chi2

| candidate | source_dir | chi2dN_total | highE_mean_absdev_first3 | highE_mean_shortfall_first3 | BNP | powerCS |
| --- | --- | --- | --- | --- | --- | --- |
| poly_bstar0_cslog2_cspower | poly_bstar0_followup_results | 0.789639 | 0.0559172 | 0.0180339 |  | 0.433905 |
| poly_bstar_bmu_cslog_cspower_blog | poly_cs_pivot_followup_results | 0.794343 | 0.0562309 | 0.0133314 |  | 0.6636 |
| poly_bstar_bmu_cslog_cspower | poly_cs_pivot_followup_results | 0.794559 | 0.0564681 | 0.0138082 |  | 0.587423 |
| poly_bstar_cslog2_cspower | poly_cs_power_followup_results | 0.796024 | 0.0525174 | 0.0152421 | 0.437402 | 0.448659 |
| poly_bstar0_cslog_cspower | poly_bstar0_followup_results | 0.797732 | 0.056743 | 0.0133416 |  | 0.680308 |
| poly_bstar_cslog_cspower_blog | poly_cs_pivot_followup_results | 0.797877 | 0.0572686 | 0.0136266 |  | 0.666472 |
| bern2_sqrtx_bstar_cslog | bernstein_shape_followup_results | 0.803125 | 0.0573703 | 0.0135908 | 3.80914 |  |
| poly_bstar_cslog_cspower | poly_cs_power_followup_results | 0.803928 | 0.0524733 | 0.0124468 | 0.815363 | 0.647617 |
| bern2_x_bstar_cslog | bernstein_shape_followup_results | 0.804816 | 0.0583396 | 0.0141585 | 3.87447 |  |
| art17m2_art23cslog_refine | art23_family_refine_results | 0.817226 | 0.0536895 | 0.0143662 | 1.95025 |  |
| art17m2_art23cslog2 | art23_family_refine_results | 0.817229 | 0.0536897 | 0.0143801 | 1.95152 |  |
| art17m2_art23cslog | art23_family_results | 0.81723 | 0.0536917 | 0.0143868 | 1.94966 |  |

## Best Plain Fits By High-Energy Absdev

| candidate | source_dir | highE_mean_absdev_first3 | highE_mean_shortfall_first3 | chi2dN_total | BNP | powerCS |
| --- | --- | --- | --- | --- | --- | --- |
| poly_bstar_cslog_loggauss | localized_shape_followup_results | 0.0512202 | 0.010737 | 0.822163 | 1.62254 |  |
| poly_bstar_cslog_loggauss_odd | localized_shape_followup_results | 0.0519633 | 0.0114563 | 0.823322 | 1.51986 |  |
| poly_cslog2_cspower | poly_cs_power_followup_results | 0.0520465 | 0.0173057 | 0.820978 | 0.690409 | 0.135394 |
| poly_cslog2_cspower_blog | poly_cs_pivot_followup_results | 0.0520609 | 0.0173866 | 0.820969 |  | 0.134199 |
| poly_bstar_bmu_cslog2_cspower | poly_cs_power_followup_results | 0.0520764 | 0.0165388 | 0.820656 | 0.87612 | 0.010597 |
| art23_mu_poly_cslog2_refine | art23_family_refine_results | 0.0522467 | 0.0157034 | 0.824755 | 0.952777 |  |
| art17_model2 | market_np_results | 0.0522816 | 0.0168682 | 0.97479 |  |  |
| poly_bstar_cslog2_refine | poly_bstar_followup_results | 0.0522845 | 0.0153166 | 0.824961 | 0.984022 |  |
| poly_bstar_bmu_cslog2 | poly_bstar_followup_results | 0.0523032 | 0.0158099 | 0.821518 | 0.948181 |  |
| poly_bstar_bmu_cslog | poly_bstar_followup_results | 0.0523698 | 0.0128263 | 0.831905 | 1.92262 |  |
| art23_mu_poly_cslog2 | art23_family_results | 0.0524514 | 0.0147339 | 0.826406 | 1.06738 |  |
| art23_mu_poly_bstar_cslog2 | art23_family_refine_results | 0.0524546 | 0.0143146 | 0.827373 | 1.13228 |  |

## Best Plain Fits By High-Energy Shortfall

| candidate | source_dir | highE_mean_shortfall_first3 | highE_mean_absdev_first3 | chi2dN_total | BNP | powerCS |
| --- | --- | --- | --- | --- | --- | --- |
| poly_bstar_cslog_loggauss | localized_shape_followup_results | 0.010737 | 0.0512202 | 0.822163 | 1.62254 |  |
| poly_bstar_cslog_loggauss_odd | localized_shape_followup_results | 0.0114563 | 0.0519633 | 0.823322 | 1.51986 |  |
| poly_bstar_cslog_cspower | poly_cs_power_followup_results | 0.0124468 | 0.0524733 | 0.803928 | 0.815363 | 0.647617 |
| art23_mu_poly_bstar_cslog | art23_family_results | 0.0124987 | 0.0527657 | 0.835585 | 1.52271 |  |
| poly_cslog_cspower | poly_cs_power_followup_results | 0.0125201 | 0.0529207 | 0.836939 | 1.41213 | 0.0530587 |
| poly_bstar_cslog_refine | poly_bstar_followup_results | 0.0126821 | 0.052718 | 0.834054 | 1.59081 |  |
| art23_mu_poly_cslog | art23_family_results | 0.0127296 | 0.0529517 | 0.837147 | 1.50055 |  |
| poly_bstar_bmu_cslog | poly_bstar_followup_results | 0.0128263 | 0.0523698 | 0.831905 | 1.92262 |  |
| art23_fi_cslog | market_np_results | 0.0128446 | 0.0542146 | 0.85637 | 1.45001 |  |
| art23_fi_cslog_refit | art23_family_results | 0.0128617 | 0.0542264 | 0.856371 | 1.45224 |  |
| poly_bstar_bmu_cslog_cspower_blog | poly_cs_pivot_followup_results | 0.0133314 | 0.0562309 | 0.794343 |  | 0.6636 |
| poly_bstar0_cslog_cspower | poly_bstar0_followup_results | 0.0133416 | 0.056743 | 0.797732 |  | 0.680308 |

## Weighted/Pareto Runs

| candidate | penalty_mode | lambda | combined_objective | chi2dN_total | highE_mean_absdev_first3 | highE_mean_shortfall_first3 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_0112_art23cs | balanced_absdev | 0 | 0.817009 | 0.81701 | 0.0559984 | 0.0147593 |
| hybrid_0112_art23cs | shortfall | 0 | 0.818626 | 0.818627 | 0.0564677 | 0.014375 |
| baseline_unfrozen | balanced_absdev | 0 | 0.838332 | 0.838333 | 0.0552647 | 0.0203656 |
| baseline_unfrozen | shortfall | 0 | 0.838612 | 0.838614 | 0.055029 | 0.0202093 |
| baseline_unfrozen | balanced_absdev | 0 | 0.838976 | 0.838976 | 0.0549653 | 0.0201175 |
| baseline_unfrozen | balanced_absdev | 0 | 0.838976 | 0.838976 | 0.0549653 | 0.0201175 |
| baseline_unfrozen | balanced_absdev | 0 | 0.838976 | 0.838976 | 0.0549653 | 0.0201175 |
| baseline_unfrozen | shortfall | 0 | 0.839328 | 0.839324 | 0.0550359 | 0.0201774 |
| baseline_0112 | shortfall | 0 | 0.839981 | 0.839983 | 0.0549476 | 0.0200868 |
| baseline_0112 | balanced_absdev | 0 | 0.839985 | 0.839989 | 0.0549522 | 0.0201025 |
| baseline_0112 | balanced_absdev | 0 | 0.839985 | 0.839989 | 0.0549522 | 0.0201025 |
| baseline_0112 | balanced_absdev | 0 | 0.839985 | 0.839989 | 0.0549522 | 0.0201025 |

The full weighted history is in the CSV linked above.
