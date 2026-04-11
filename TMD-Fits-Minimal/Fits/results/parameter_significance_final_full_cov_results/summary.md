# Final Full-Covariance Significance

- card: `Final.jl`
- card chi2/N: `1.119776`
- fitted chi2/N: `1.119701`
- delta chi2 total from card to fitted point: `-0.034628`
- chi2 total: `520.660992`
- local refit evals: `303`
- local refit flag: `0`
- hessian evals: `163`
- grad norm (log10 objective): `0.0289438`
- hessian min eig: `-48173.4`
- hessian max eig: `1.6401e+06`
- max |corr|: `0.784394`

Top correlations:

- `lambda2` vs `lambda3`: `-0.784394`
- `lambda2` vs `amp`: `-0.769596`
- `lambda3` vs `amp`: `0.749726`
- `logx0` vs `c1`: `-0.701824`
- `lambda1` vs `c1`: `-0.606594`
- `lambda1` vs `c0`: `0.547747`
- `logx0` vs `sigx`: `-0.494273`
- `lambda1` vs `sigx`: `0.477549`

Most constrained parameters by |z vs reference|:

- `c0`: value `0.0665399`, sigma `259.595`, z `0.000`
- `lambda2`: value `0.988183`, sigma `4162.69`, z `0.000`
- `lambda3`: value `-2.04018`, sigma `13410.1`, z `-0.000`
- `c1`: value `0.0301422`, sigma `254.632`, z `0.000`
- `amp`: value `-0.276601`, sigma `4996.99`, z `-0.000`
- `lambda1`: value `0.00300845`, sigma `424.349`, z `0.000`

Top nested tests by Delta chi2:

- `c0_zero`: Delta chi2 `333.877130`, nested chi2/N `1.837716`
- `c1_zero`: Delta chi2 `68.063624`, nested chi2/N `1.266074`
- `lambda2_zero`: Delta chi2 `57.923618`, nested chi2/N `1.244268`
- `lambda3_zero`: Delta chi2 `46.951491`, nested chi2/N `1.220672`
- `amp_zero`: Delta chi2 `24.133378`, nested chi2/N `1.171601`
- `bump_off`: Delta chi2 `17.731571`, nested chi2/N `1.157833`
- `lambda1_zero`: Delta chi2 `-0.001427`, nested chi2/N `1.119698`