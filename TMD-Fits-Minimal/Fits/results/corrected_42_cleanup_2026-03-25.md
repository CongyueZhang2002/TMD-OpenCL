# Corrected 4-2 Cleanup Record (2026-03-25)

This records the second cleanup pass on the corrected `4-2` exploration branch.

Scope:
- keep the current chosen broad-bump model live
- keep one no-bump `4-2` reference live
- archive the extra corrected `4-2` alternatives that are no longer intended for
  active use

## Kept Live

| Card | chi2/N | highE absdev | highE shortfall | Reason |
| --- | ---: | ---: | ---: | --- |
| `BroadBump42LogGaussAlpha1NoLambda2.jl` | 1.007120 | 0.053682 | 0.010805 | Current chosen corrected `4-2` model |
| `Art23FamilyMuPolyBstarCSLog42Best.jl` | 1.029354 | 0.051938 | 0.010810 | No-bump `4-2` reference kept for comparison |

## Archived Cards

| Card | chi2/N | highE absdev | highE shortfall | Reason |
| --- | ---: | ---: | ---: | --- |
| `BroadBump42FixedGaussBasisBest.jl` | 1.013174 | 0.052111 | 0.009116 | Fixed-shape bump requires hard-coded center/width and is not intended as the active paper-facing parameterization |
| `PolyBstarCSLogLogGauss42BroadSig.jl` | 0.995503 | 0.050966 | 0.009354 | Better metrics than the chosen model, but remained highly correlated and still produced structured small-`x` TMD shapes |

## Archived Kernels

| Kernel | Reason |
| --- | --- |
| `NP-BroadBump42FixedGaussBasis.cl` | Only used by the archived fixed-shape corrected `4-2` card |
| `NP-BroadBump42FixedGaussBasis.jl` | Only used by the archived fixed-shape corrected `4-2` card |
| `NP-PolyBstarCSLogLogGauss42Best.cl` | Only used by archived sharp-family corrected `4-2` cards |
| `NP-PolyBstarCSLogLogGauss42Best.jl` | Only used by archived sharp-family corrected `4-2` cards |

## Archived Result Bundles

| Path | Reason |
| --- | --- |
| `Fits/corrected_42_followup_results` | Broad-sig and fixed-shape follow-up refits only |
| `Fits/corrected_42_variant_refit_results` | Fixed-shape corrected refit and older sharp corrected refresh only |
| `Fits/bump_reparam_42_results` | Fixed-shape bump-reparameterization search outputs only |
| `Fits/parameter_significance_broad_bump_42_fixed_gauss_basis_corrected_results` | Fixed-shape significance outputs only |
| `Fits/parameter_significance_fixed_gauss_basis_42_results` | Older fixed-shape significance outputs only |
| `Fits/parameter_significance_poly_bstar_cslog_loggauss_42broadsig_results` | Broad-sig significance outputs only |
| `Fits/bump_reparam_42_search.py` | Search driver for the archived fixed-shape branch only |
| `Fits/corrected_42_followup_refit.py` | Driver for archived follow-up branch only |
| `Fits/deep_refit_corrected_42_variants.py` | Driver for archived corrected-variant branch only |

No files are deleted here; archived items are moved into `deprecated/corrected_42`
subfolders.
