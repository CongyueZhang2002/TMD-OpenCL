# Recent 4-2 Parameterization Record (2026-03-24)

This records the recent 4-2 cleanup shortlist before removing generated clutter.

| Label | Card | chi2/N | highE absdev | highE shortfall | max |corr| | Status | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Sharp bump 4-2 best | `PolyBstarCSLogLogGauss42Best.jl` | 0.995536 | 0.052741 | 0.009500 |  | keep | Best sharp-bump / shortfall-oriented 4-2 representative |
| No-bump 4-2 best | `Art23FamilyMuPolyBstarCSLog42Best.jl` | 1.029354 | 0.051938 | 0.010810 |  | keep | Best no-bump 4-2 representative |
| Broad bump alpha=1 best | `BroadBump42LogGaussAlpha1Best.jl` | 0.990112 | 0.051888 | 0.009656 |  | drop | Superseded by the no-lambda2 broad-bump refit |
| Current broad-bump best | `BroadBump42LogGaussAlpha1NoLambda2.jl` | 0.988885 | 0.052318 | 0.009853 | 0.970 | keep | Main current best 4-2 model |
| Current broad-bump no lambda4 | `BroadBump42LogGaussAlpha1NoLambda2NoLambda4Aggressive.jl` | 0.989024 | 0.054120 | 0.011499 |  | drop | Comparable total chi2 but clearly worse high-energy metrics |
| xbar+quad+logpair broad bump | `BroadBump42XbarQuadLogPair.jl` | 0.988827 | 0.052430 | 0.009945 |  | drop | Interesting alternative basis but not cleaner than kept representatives |
| Lorentz bump swap | `BroadBump42SwapBumpLorentz.jl` | 0.988511 | 0.052252 | 0.010547 |  | drop | Tiny chi2 gain but much worse conditioning |
| Area-normalized Gaussian bump | `BroadBump42AreaGauss.jl` | 0.988217 | 0.053031 | 0.010701 |  | drop | Slight chi2 gain only; high-energy metrics worse |
| Area-normalized Lorentz bump | `BroadBump42AreaLorentz.jl` | 0.989498 | 0.052554 | 0.010163 |  | drop | No clear gain over current broad-bump best |
| Fixed broad Gaussian basis | `BroadBump42FixedGaussBasisBest.jl` | 0.992602 | 0.051445 | 0.009108 | 0.742 | keep | Cleaner paper-facing alternative; better high-energy metrics and lower correlation |

Kept shortlist:
- `PolyBstarCSLogLogGauss42Best.jl`: Best sharp-bump / shortfall-oriented 4-2 representative
- `Art23FamilyMuPolyBstarCSLog42Best.jl`: Best no-bump 4-2 representative
- `BroadBump42LogGaussAlpha1NoLambda2.jl`: Main current best 4-2 model
- `BroadBump42FixedGaussBasisBest.jl`: Cleaner paper-facing alternative; better high-energy metrics and lower correlation

Dropped entries are still represented in the CSV above, but their generated cards / kernels / raw result folders can be removed safely.