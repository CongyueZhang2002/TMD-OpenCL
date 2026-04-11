# Bayesian Parameter Extraction

This folder contains a small extractor for the Bayesian MCMC backends stored in the source repo's `TMD-Fits-Minimal/Fits/Bayes_Data/` area, plus locally downloaded source chain files and generated CSV output.

## What Was Inspected

Source repo:
- `TMD-Fits-Minimal/Fits/Bayes_Data/`
- `TMD-Fits-Minimal/Fits/fit_bayesian.ipynb`
- `TMD-Fits-Minimal/Fits/fit_replicas.ipynb`
- `TMD-Fits-Minimal/Cards/BroadBump42LogGaussAlpha1NoLambda2.jl`
- `TMD-Fits-Minimal/Cards/Final.jl`

Current target repo:
- `TMD-Fits-Minimal/Fits/replicas.csv`
- `TMD-Fits-Minimal/Fits/fit.ipynb`
- `TMD-Fits-Minimal/Fits/fit_replicas.ipynb`
- `TMD-Fits-Minimal/Cards/Final.jl`
- `TMD-Fits-Minimal/TMDs/NP Parameterizations Julia/NP-BroadBump42LogGaussAlpha1NoLambda2.jl`

The current target repo's top-level `replicas.csv` uses an older, different parameter convention (`g2`, `l`, `l2`, `N1`, ...). That does not match the inspected source Bayesian `9d` model, so this extractor uses the verified source-card order for the Bayesian output instead of trying to force that older CSV header.

## Verified Source Findings

The source notebook stores chains with `emcee.backends.HDFBackend` under `mcmc/chain` and `mcmc/log_prob`. The inspected source backends do not expose `mcmc/blobs`.

For the committed source notebook/card pair:
- `burn = 1000`
- `thin = 1`
- flattening is done across walkers via `backend.get_chain(discard=burn, flat=True, thin=thin)`
- the verified 9-parameter model is `BroadBump42LogGaussAlpha1NoLambda2`
- the verified 9-parameter order is:
  `lambda1, lambda2, lambda3, logx0, sigx, amp, BNP, c0, c1`
- those chain coordinates are stored in normalized `[0, 1]` units and are converted back to physical values with `LOWER + theta * (UPPER - LOWER)` using the source card bounds

## 6d vs 9d

The source `Bayes_Data/` directory contains both `6d` and `9d` backends.

The committed source notebook/card cleanly verify the `9d` mapping above. The source `6d` files are more ambiguous:
- the committed code path constructs `ndim = len(free_idx_replica)` from the card's `frozen_indices`
- the committed card has `frozen_indices = []`, which implies `9d`
- the saved notebook outputs that mention `6d` are inconsistent with the committed code and even show labels like `p10`, so they clearly came from an older execution state

Because of that, the extractor does not invent a built-in `6d` parameter map. For `6d` chains, pass an explicit JSON parameter spec with the sampled parameter names and bounds.

## Downloaded Source Files

Downloaded Bayes_Data files live in:
- `TMD-Fits-Minimal/Bayesian Analysis/source_data/`

At the moment this folder contains the real source backend used for the generated CSV:
- `source_data/emcee_gated_global_9d.h5`

## Generated Output

Generated CSV files are written directly into:
- `TMD-Fits-Minimal/Bayesian Analysis/`

Included here:
- `bayesian_parameters_emcee_gated_global_9d.csv`

The included CSV was regenerated successfully from the real source backend with:
- 384000 rows
- 13 columns
- `walker` values `0..63`
- `step` values `1000..6999`

Columns are:
- `sample_id`
- `walker`
- `step`
- `log_prob` when available
- then the named fit parameters in chain order

`chi2dN` is intentionally not written. In the inspected source notebook, `log_prob` is a posterior quantity built from a Beta prior plus a PDF-marginalized likelihood, so `chi2dN` is not recoverable from the stored backend without rerunning the full model path.

## Exact Commands

Download the same real source backend used for the included CSV:

```powershell
python "TMD-Fits-Minimal/Bayesian Analysis/download_bayes_data.py" emcee_gated_global_9d.h5
```

Regenerate the included 9d CSV with the source notebook defaults:

```powershell
python "TMD-Fits-Minimal/Bayesian Analysis/extract_bayesian_parameters.py" "TMD-Fits-Minimal/Bayesian Analysis/source_data/emcee_gated_global_9d.h5"
```

Override burn-in or thinning if needed:

```powershell
python "TMD-Fits-Minimal/Bayesian Analysis/extract_bayesian_parameters.py" `
  --burn 1000 `
  --thin 1 `
  "TMD-Fits-Minimal/Bayesian Analysis/source_data/emcee_gated_global_9d.h5"
```

Example custom-spec flow for an ambiguous `6d` chain:

```json
{
  "parameterization": "custom_6d_example",
  "chain_storage": "normalized",
  "parameters": [
    {"name": "param_a", "lower": 0.0, "upper": 1.0},
    {"name": "param_b", "lower": 0.0, "upper": 1.0},
    {"name": "param_c", "lower": 0.0, "upper": 1.0},
    {"name": "param_d", "lower": 0.0, "upper": 1.0},
    {"name": "param_e", "lower": 0.0, "upper": 1.0},
    {"name": "param_f", "lower": 0.0, "upper": 1.0}
  ]
}
```

```powershell
python "TMD-Fits-Minimal/Bayesian Analysis/extract_bayesian_parameters.py" `
  --parameter-spec path\\to\\custom_6d_spec.json `
  "TMD-Fits-Minimal/Bayesian Analysis/source_data/emcee_gated_global_6d.h5"
```

## Source References

Source notebook:
- https://github.com/CZhou-24/TMD-CUDA-main/blob/main/TMD-Fits-Minimal/Fits/fit_bayesian.ipynb

Source card:
- https://github.com/CZhou-24/TMD-CUDA-main/blob/main/TMD-Fits-Minimal/Cards/BroadBump42LogGaussAlpha1NoLambda2.jl

## Additional Notes

- `emcee_truth_global_9d.h5` exists in the source repo, but it contains exactly 1000 steps. With the source notebook's `burn = 1000`, that leaves zero post-burn samples, so the extractor correctly errors unless you intentionally lower `--burn`.
- The source `6d` backends are supported structurally by the extractor, but only through an explicit user-supplied JSON parameter spec. No built-in `6d` map is shipped because the committed source notebook/card do not verify one consistently.
