# Math Modelling Project - Basics of Modelling the Pedestrian Flow

This repository contains a Python reproduction scaffold for the single-file
pedestrian-flow model from:

> A. Seyfried, B. Steffen, T. Lippert,  
> **"Basics of Modelling the Pedestrian Flow"**,  
> *Physica A* **368** (2006) 232-238.  
> arXiv: [physics/0506189](https://arxiv.org/abs/physics/0506189)

The code simulates pedestrians on a one-dimensional ring, reproduces the
velocity-density relation ("fundamental diagram"), compares the paper's two
interaction variants, and includes diagnostics plus appendix-style sensitivity
experiments.

## What Is Implemented

The package models single-file pedestrian motion with periodic boundary
conditions. For a ring of length $L$ with $N$ pedestrians, the density is

$$
\rho = \frac{N}{L}.
$$

Pedestrian $i$ reacts only to the pedestrian directly in front. Mass is set to
$m_i = 1$, so force and acceleration use the same numerical value.

The shared equation of motion is:

$$
\frac{dx_i}{dt} = v_i,
$$

$$
\frac{dv_i}{dt}
= F_i
= \frac{v_i^0 - v_i}{\tau} + F_i^{\mathrm{int}}.
$$

The velocity-dependent required length is:

$$
d_i = a + b \max(v_i, 0).
$$

Default paper parameters are stored in `ModelParameters`:

| parameter | default | meaning |
| --- | ---: | --- |
| `a` | `0.36` m | required-length offset |
| `b` | `0.56` s | velocity dependence of required length |
| `tau` | `0.61` s | acceleration/relaxation time |
| `v0_mean` | `1.24` m/s | mean intended speed |
| `v0_sigma` | `0.05` m/s | intended-speed standard deviation |
| `e` | `0.07` | remote-action strength |
| `f` | `2.0` | remote-action exponent |
| `clear_distance_floor` | `1e-6` m | numerical guard for Eq. 6 |

Two model classes are available:

- `HardBodyModel`: hard bodies without remote action, corresponding to Eq. 5.
  Moves that violate the required length are rejected and relaxed iteratively.
- `RemoteActionModel`: hard bodies with remote action, corresponding to Eq. 6.
  A repulsive force is integrated with explicit Euler.

For `HardBodyModel`, the interaction is implemented as a hard constraint:

$$
F_i =
\begin{cases}
\dfrac{v_i^0 - v_i}{\tau}, & \text{if } g_i > d_i, \\
0 \text{ and rejected move}, & \text{if } g_i \le d_i,
\end{cases}
$$

where $g_i$ is the gap to the pedestrian in front.

For `RemoteActionModel`, the remote repulsion is:

$$
G_i = \frac{v_i^0 - v_i}{\tau} - e \left(\frac{1}{g_i - d_i}\right)^f.
$$

$$
F_i =
\begin{cases}
G_i, & \text{if } v_i > 0, \\
\max(0, G_i), & \text{if } v_i \le 0.
\end{cases}
$$

## Repository Layout

```text
Math_Modelling/
├── pedestrian/
│   ├── __init__.py
│   ├── model.py                 # ModelParameters, HardBodyModel, RemoteActionModel
│   ├── simulation.py            # run_single and RunResult
│   ├── fundamental_diagram.py   # density sweeps
│   └── empirical.py             # empirical CSV helpers and RMSE utilities
├── scripts/
│   ├── _common.py
│   ├── run_figure1.py           # Fig. 1: hard-body fundamental diagram
│   ├── run_figure2.py           # Fig. 2: remote action comparison
│   ├── run_figure3.py           # Fig. 3: space-time density waves
│   ├── validate_reproduction.py # quantitative diagnostics
│   └── run_appendix_experiments.py
├── data/
│   ├── README.md
│   └── empirical_single_file_points.csv
├── tests/
│   └── test_model.py
├── demo/
│   ├── demo.py                  # self-contained HTML replay generator
│   └── demo_simulation.html     # generated demo replay
├── appendix_experiment_results/ # generated/checked-in appendix outputs
├── figures/                     # generated paper figure outputs
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The project is not packaged with `setup.py` or `pyproject.toml`; run commands
from the repository root. The `.venv/` directory is local only and does not
need to be submitted; recreate it with the commands above.

## Quick Reproduction

These commands are the shortest path for checking that the repository works
from a clean copy:

```bash
pytest -q

python scripts/run_figure1.py --preset quick --no-show
python scripts/run_figure2.py --preset quick --no-show
python scripts/run_figure3.py --preset quick --no-show

python demo/demo.py --table
```

Expected outputs after the figure commands:

```text
figures/figure1_hardbody.png
figures/figure2_remote.png
figures/figure3_density_waves.png
figures/figure3_density_waves.pdf
```

`--preset quick` is intended for fast verification. Use `--preset default` for
cleaner curves and `--preset paper` for the expensive setting closest to the
published simulation length.

## Main Commands

Run the tests:

```bash
pytest -q
```

Reproduce the figure outputs with the default preset:

```bash
python scripts/run_figure1.py --no-show
python scripts/run_figure2.py --no-show
python scripts/run_figure3.py --no-show
```

The figure scripts share these step-count presets:

```text
quick   = 20_000 relaxation + 20_000 measurement steps
default = 60_000 relaxation + 60_000 measurement steps
paper   = 300_000 relaxation + 300_000 measurement steps
```

Example:

```bash
python scripts/run_figure1.py --preset paper --no-show
```

Run quantitative diagnostics:

```bash
python scripts/validate_reproduction.py --preset quick
python scripts/validate_reproduction.py --preset paper --json
```

Run the appendix sensitivity experiments:

```bash
python scripts/run_appendix_experiments.py --preset quick
python scripts/run_appendix_experiments.py --preset quick --output-dir appendix_experiment_results
```

This writes CSV, JSON, and PNG summaries for:

- `A_b_sensitivity`
- `B_a_sensitivity`
- `C_tau_sensitivity`
- `C_tau_transient`
- `D_N_rho`
- `E_finite_size`

## HTML Demo

Generate a self-contained browser replay:

```bash
python demo/demo.py
```

By default this writes:

```text
demo/demo_simulation.html
```

Open it automatically after generation:

```bash
python demo/demo.py --show
```

Print the older numeric smoke-test table instead of writing HTML:

```bash
python demo/demo.py --table
```

Useful demo overrides:

```bash
python demo/demo.py --frames 120 --relax-steps 4000 --fd-relax-steps 1500 --fd-measure-steps 1500
python demo/demo.py --output /tmp/demo_simulation.html
```

## Use As A Library

```python
from pedestrian import ModelParameters, HardBodyModel, fundamental_diagram

params = ModelParameters(a=0.36, b=0.56)
results = fundamental_diagram(
    HardBodyModel,
    params,
    L=17.3,
    densities=[0.5, 1.0, 1.5, 2.0, 2.5],
)

for result in results:
    print(result.density, result.mean_velocity, result.std_velocity)
```

Run one simulation directly:

```python
from pedestrian import ModelParameters, RemoteActionModel, run_single

result = run_single(
    RemoteActionModel,
    n=21,
    L=17.3,
    params=ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0),
    relax_steps=20_000,
    measure_steps=20_000,
    seed=0,
)

print(result)
```

## Data

`data/empirical_single_file_points.csv` stores 170 empirical marker positions
digitized from the paper's Fig. 1 EPS source. These are plotted data points,
not a fitted empirical curve. See `data/README.md` for the coordinate mapping
used during digitization.

## Reproduction Notes

- $b = 0.56\,\mathrm{s}$ in `HardBodyModel` is the main setting that reproduces the
  empirical fundamental diagram shape.
- $b = 0$ with `RemoteActionModel` shows the velocity gap and stop-and-go
  density waves near $\rho \approx 1.2\,\mathrm{m}^{-1}$.
- `run_figure3.py` produces a space-time position plot, not a
  velocity-density diagram.
- The expensive paper preset uses $300{,}000 + 300{,}000$ integration steps
  per density point with $\Delta t = 0.001\,\mathrm{s}$.
- Figure PNGs are generated artifacts; `.gitignore` ignores `figures/*.png`.
