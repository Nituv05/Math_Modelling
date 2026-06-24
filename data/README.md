# Data

`empirical_single_file_regression.csv` is a reference curve, not raw
experimental data.  It is generated from the single-file regression reported
by Seyfried, Steffen, Klingsch and Boltes in "The Fundamental Diagram of
Pedestrian Movement Revisited" (arXiv:physics/0506170):

```text
1 / rho = 0.36 m + 1.06 s * v
```

The modelling paper (`physics/0506189`) uses this relation as the empirical
comparison for the fundamental diagram.  Use this CSV for overlays and
sanity checks.  Do not treat it as a replacement for the original measured
individual observations.
