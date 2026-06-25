# Math Modelling Project — Basics of Modelling the Pedestrian Flow

A Python reproduction and validation scaffold for

> A. Seyfried, B. Steffen, T. Lippert,
> **"Basics of Modelling the Pedestrian Flow"**,
> *Physica A* **368** (2006) 232–238.
> arXiv: [physics/0506189](https://arxiv.org/abs/physics/0506189)

The paper studies single-file (one-dimensional) pedestrian motion with a
**modified social-force model** and asks which microscopic interaction
reproduces the empirical **velocity–density relation** (fundamental
diagram). It compares two interaction approaches and shows that a
*velocity-dependent required length* `d = a + b·v` is the key ingredient.

---

## The model

Pedestrians move on a ring of length `L` (periodic boundary conditions),
ordered as `x₁ < x₂ < … < x_N`. Each pedestrian `i` only reacts to the
person `i+1` directly in front. Masses are set to `mᵢ = 1`.

**Equation of motion (Eq. 1–2)**

```
dxᵢ/dt = vᵢ
dvᵢ/dt = Fᵢ = (vᵢ⁰ − vᵢ)/τ  +  (repulsion)
```

**Required length (Eq. 4)** — personal space grows with speed:

```
d = a + b·v        a = 0.36 m,   b ∈ {0, 0.56, 1.06} s
```

### 1. Hard bodies *without* remote action (Eq. 5) — `HardBodyModel`

```
Fᵢ = (vᵢ⁰ − vᵢ)/τ           if  gapᵢ > dᵢ
     velocity → 0, no move   if  gapᵢ ≤ dᵢ   (hard collision)
```

Integrated with the special quasi-parallel update of Section II C
(explicit Euler step `Δt = 0.001 s`, then reject moves that violate the
required length and relax to a fixpoint).

### 2. Hard bodies *with* remote action (Eq. 6) — `RemoteActionModel`

```
Gᵢ = (vᵢ⁰ − vᵢ)/τ − e·(1 / (gapᵢ − dᵢ))^f
Fᵢ = Gᵢ              if vᵢ > 0
Fᵢ = max(0, Gᵢ)      if vᵢ ≤ 0          e = 0.07 N,  f = 2
```

Integrated with explicit Euler, `Δt = 0.001 s`.

### Parameters (Section III)

| symbol | value | meaning |
|--------|-------|---------|
| `a` | 0.36 m | required-length offset |
| `b` | 0 / 0.56 / 1.06 s | velocity dependence of required length |
| `τ` | 0.61 s | acceleration time constant |
| `v⁰` | N(1.24, 0.05) m/s | intended speed (normal distribution) |
| `e`, `f` | 0.07 N, 2 | remote-force strength / range |
| `L` | 17.3 m | ring length (20, 50 m give the same results) |
| `Δt` | 0.001 s | time step |
| steps | 3·10⁵ + 3·10⁵ | relaxation + measurement |

Initial condition: `t = 0`, all velocities zero, persons placed randomly
with minimal distance `a`.

---

## Project layout

```
Math_Modelling/
├── pedestrian/                  # the model package
│   ├── model.py                # HardBodyModel, RemoteActionModel, parameters
│   ├── simulation.py           # run one simulation, measure mean velocity
│   ├── fundamental_diagram.py  # sweep density -> v(ρ)
│   └── empirical.py            # empirical data-point helpers
├── scripts/
│   ├── run_figure1.py          # Fig. 1: hard bodies, b = 0 / 0.56 / 1.06
│   ├── run_figure2.py          # Fig. 2: remote action vs. hard bodies
│   ├── run_figure3.py          # Fig. 3: space-time density waves, Eq. 6 b=0
│   └── validate_reproduction.py # quantitative validation diagnostics
├── data/
│   └── empirical_single_file_points.csv
├── demo.py                     # animated browser demo
├── tests/test_model.py         # invariants & qualitative checks (pytest)
├── figures/                    # generated PNGs
├── requirements.txt
└── README.md
```

---

## Usage

```bash
pip install -r requirements.txt

# Reproduce the figures (saved into figures/).
python scripts/run_figure1.py --no-show          # fundamental diagram, Eq. 5
python scripts/run_figure2.py --no-show          # remote action, Eq. 6
python scripts/run_figure3.py --no-show          # space-time density waves, Eq. 6 b=0

# Step-count presets: --preset quick | default | paper
#   quick   = 20k+20k   (fast smoke test)
#   default = 60k+60k   (reproduces the shape)
#   paper   = 300k+300k (the published runs; slow)
python scripts/run_figure1.py --preset paper --no-show

# Run an animated simulation demo (saved as figures/demo_simulation.html)
python demo.py

# Optional: print the old numeric smoke-test table
python demo.py --table

# Run quantitative diagnostics against nearby empirical data points,
# finite-size checks, remote-action velocity gap, and force-floor sensitivity
python scripts/validate_reproduction.py --preset quick
python scripts/validate_reproduction.py --preset paper --json
```

Run the test suite:

```bash
pytest -q
```

### Use the model directly

```python
from pedestrian import ModelParameters, HardBodyModel, fundamental_diagram

params = ModelParameters(a=0.36, b=0.56)
results = fundamental_diagram(HardBodyModel, params, L=17.3,
                             densities=[0.5, 1.0, 1.5, 2.0, 2.5])
for r in results:
    print(r.density, r.mean_velocity)
```

---

## Key result reproduced

* With a **velocity-independent** required length (`b = 0`) the model gives a
  fundamental diagram with the *wrong* curvature.
* A **velocity-dependent** required length with **`b = 0.56 s`** reproduces
  the empirical fundamental diagram well (Fig. 1).
* The **remote action** (Eq. 6) has only a small influence when `b > 0`, but
  for `b = 0` it produces a **velocity gap and stop-and-go density waves**
  near `ρ ≈ 1.2 1/m` (Figs. 2 & 3).
* **Fig. 3 is not an empirical-data plot.** It is a space-time plot of the
  simulated positions for the `RemoteActionModel` with `b = 0`, comparing
  densities just below and just above the Fig. 2 velocity gap.

---

## Notes on the reproduction

* The published figures use 3·10⁵ + 3·10⁵ steps at `Δt = 0.001 s` for every
  density point. The scripts default to a lighter preset that already shows
  the correct shape; pass `--preset paper` for the full runs.
* The hard-body update (Eq. 5) uses iterative relaxation of the rejected
  moves, which is the "approximation to the exact parallel update" the
  authors describe in Section II C.
* The empirical overlay is a set of data points digitized from the Fig. 1 EPS
  source. The paper does not plot an empirical fitted curve in Figs. 1-3;
  Eq. 4 is the required-length relation used for the model parameter `d`.
* Fig. 3 corresponds to the two space-time panels included from the paper
  source as `fig3.eps` and `fig4.eps`: `rho ~= 1.16 1/m` and
  `rho ~= 1.21 1/m`. It is meant to explain where the velocity gap in Fig. 2
  comes from.
* `RemoteActionModel` uses explicit Euler consistently: force is evaluated
  at the old state, velocity is advanced, and position is advanced with the
  old velocity.
