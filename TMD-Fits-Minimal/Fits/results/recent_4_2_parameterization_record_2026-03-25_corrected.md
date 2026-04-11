# Corrected 4-2 Parameterization Record (2026-03-25)

This records the corrected `4-2` shortlist after restoring the proper `\mu_*`
behavior and before archiving superseded exploration files.

## Kept Representatives

| Card | chi2/N | highE absdev | highE shortfall | Reason |
| --- | ---: | ---: | ---: | --- |
| `BroadBump42LogGaussAlpha1NoLambda2.jl` | 1.007120 | 0.053682 | 0.010805 | Current chosen broad-bump `4-2` model |
| `BroadBump42FixedGaussBasisBest.jl` | 1.013174 | 0.052111 | 0.009116 | Cleaner fixed-shape comparison model |
| `Art23FamilyMuPolyBstarCSLog42Best.jl` | 1.029354 | 0.051938 | 0.010810 | Best no-bump `4-2` representative |
| `PolyBstarCSLogLogGauss42BroadSig.jl` | 0.995503 | 0.050966 | 0.009354 | Best broad-sig sharp-family comparison |

## Archived Cards

| Card | chi2/N | highE absdev | highE shortfall | Reason |
| --- | ---: | ---: | ---: | --- |
| `PolyBstarCSLogLogGauss42Best.jl` | 0.988362 | 0.053762 | 0.010487 | Older narrow-bump sharp model, superseded by `PolyBstarCSLogLogGauss42BroadSig.jl` |
| `BroadBump42FixedGaussBasisLog1mX.jl` | 1.012976 | 0.052064 | 0.009148 | Extra `\log(1-x)` freedom was nearly unused and did not materially improve the fixed-basis model |
| `BroadBump42FixedGaussBasisBest__SignificanceTmp.jl` |  |  |  | Temporary significance worker card |
| `BroadBump42LogGaussAlpha1NoLambda2__SignificanceTmp.jl` |  |  |  | Temporary significance worker card |
| `PolyBstarCSLogLogGauss42BroadSig__SignificanceTmp.jl` |  |  |  | Temporary significance worker card |

## Archived Kernels

| Kernel | Reason |
| --- | --- |
| `NP-BroadBump42FixedGaussBasisLog1mX.cl` | No kept card depends on it after archiving the matching card |
| `NP-BroadBump42FixedGaussBasisLog1mX.jl` | No kept card depends on it after archiving the matching card |
| `NP-BroadBump42LogGaussW060Alpha1.cl` | Superseded by the dedicated `NP-BroadBump42LogGaussAlpha1NoLambda2.cl` kernel |
| `NP-BroadBump42LogGaussW060Alpha1.jl` | Superseded by the dedicated `NP-BroadBump42LogGaussAlpha1NoLambda2.jl` kernel |

No files are deleted here; archived items are moved to the existing `deprecated`
folders.
