# Data

`empirical_single_file_points.csv` contains the empirical marker positions
shown in Fig. 1 of the modelling paper (`physics/0506189`).

The arXiv source for the paper contains the generated EPS figure, not a
separate numerical table.  The CSV was digitized directly from the Fig. 1
EPS marker coordinates using the figure axes:

```text
rho = (x - 1020) / 1400
v   = (y - 600) / 3000
```

These are plotted data points.  They should not be replaced by the linear
required-length relation from Eq. 4, which is a model parameter relation used
for `d = a + b v`, not the empirical overlay in the three figures.
