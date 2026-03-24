# Selected Parameterizations (2026-03-22)

These are the generated candidates retained after cleanup.

## 0112 fitted baseline
- Candidate: `baseline_unfrozen`
- Reason: Reference fitted 0112 baseline used in comparisons
- `chi2/N_total`: 0.835177
- `highE_mean_absdev_first3`: 0.055019
- `highE_mean_shortfall_first3`: 0.020125
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\AutoBaselineUnfrozen.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-AutoBaselineUnfrozen.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\art23_family_results\baseline_unfrozen.json`

## Poly-x bstar + CSlog
- Candidate: `art23_mu_poly_bstar_cslog`
- Reason: Representative ART23-kernel base model and prior visual favorite
- `chi2/N_total`: 0.835585
- `highE_mean_absdev_first3`: 0.052766
- `highE_mean_shortfall_first3`: 0.012499
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\Art23FamilyMuPolyBstarCSLog.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-Art23FamilyMuPolyBstarCSLog.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\art23_family_results\art23_mu_poly_bstar_cslog.json`

## Poly-x bstar + CSlog + loggauss
- Candidate: `poly_bstar_cslog_loggauss`
- Reason: Best current high-energy normalization model
- `chi2/N_total`: 0.822163
- `highE_mean_absdev_first3`: 0.051220
- `highE_mean_shortfall_first3`: 0.010737
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\PolyBstarCSLogLogGauss.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-PolyBstarCSLogLogGauss.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\localized_shape_followup_results\poly_bstar_cslog_loggauss.json`

## Poly-x bstar + CSlog alpha=1
- Candidate: `poly_bstar_cslog_alpha1`
- Reason: Reduced plain-CSlog variant accepted under the 1% chi2 worsening rule
- `chi2/N_total`: 0.837169
- `highE_mean_absdev_first3`: 0.052963
- `highE_mean_shortfall_first3`: 0.012512
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\Art23FamilyMuPolyBstarCSLogAlpha1.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-Art23FamilyMuPolyBstarCSLogAlpha1.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\pruned_model_refits_0_2_results\poly_bstar_cslog_alpha1.json`

## Poly-x bstar + CSlog + loggauss reduced
- Candidate: `poly_bstar_cslog_loggauss_reduced`
- Reason: Reduced normalization-focused variant accepted under the 1% chi2 worsening rule
- `chi2/N_total`: 0.827576
- `highE_mean_absdev_first3`: 0.051453
- `highE_mean_shortfall_first3`: 0.010487
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\PolyBstarCSLogLogGaussReduced.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-PolyBstarCSLogLogGaussReduced.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\pruned_model_refits_0_2_results\poly_bstar_cslog_loggauss_lambda3_0_alpha1.json`

## Poly-x bstar + CSlog + powerCS
- Candidate: `poly_bstar_cslog_cspower`
- Reason: Best undershoot-focused representative from the powerCS branch
- `chi2/N_total`: 0.803928
- `highE_mean_absdev_first3`: 0.052473
- `highE_mean_shortfall_first3`: 0.012447
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\PolyBstarCSLogCSPower.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-PolyBstarCSLogCSPower.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\poly_cs_power_followup_results\poly_bstar_cslog_cspower.json`

## Poly-x + CSlog2 + powerCS
- Candidate: `poly_cslog2_cspower`
- Reason: Best non-bstar powerCS normalization representative
- `chi2/N_total`: 0.820978
- `highE_mean_absdev_first3`: 0.052047
- `highE_mean_shortfall_first3`: 0.017306
- Card: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Cards\PolyCSLog2CSPower.jl`
- NP file: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\TMDs\NP Parameterizations\NP-PolyCSLog2CSPower.cl`
- Result JSON: `c:\Users\congyue zhang\Desktop\OpenCL fitter\TMD-Fits-Minimal\Fits\poly_cs_power_followup_results\poly_cslog2_cspower.json`
