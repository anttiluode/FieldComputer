# FieldComputer: measured result

This tested adaptation of the archived field retained readable history and performed nonlinear delayed-parity classification. Its incremental response to a common probe also carried history information. No comparative advantage for reciprocal coupling was established. The driven field and uncoupled nonlinear cells matched its passive scores; spatial interactions and scout feedback were unnecessary on these tasks.

## Gates

| Gate | Result | Claim tested |
|---|---|---|
| G0 | PASS | Numerical archive fidelity and finite runs |
| G1 | PASS | Readable first-pulse memory |
| G2 | PASS | Nonlinear delayed parity |
| G3 | PASS | History-conditioned incremental probe response |
| G4 | FAIL | Advantage over alternative computers with the same reader family |

## Held-out accuracy

Means across environments; ranges are the minimum and maximum seed accuracy. Chance is 50%.

| Computer | Observation | Reader | First-pulse sign | Delayed parity |
|---|---|---|---:|---:|
| State-matched delay line + PCA | passive | linear | 100.0% (100.0%–100.0%) | 50.4% (48.4%–52.3%) |
| State-matched delay line + PCA | passive | quadratic | 99.8% (99.6%–100.0%) | 100.0% (100.0%–100.0%) |
| Driven field | incremental | linear | 94.8% (88.3%–97.3%) | 88.1% (84.0%–92.6%) |
| Driven field | incremental | quadratic | 98.0% (91.8%–100.0%) | 97.7% (95.7%–99.6%) |
| Driven field | passive | linear | 100.0% (100.0%–100.0%) | 99.5% (98.8%–100.0%) |
| Driven field | passive | quadratic | 100.0% (100.0%–100.0%) | 100.0% (100.0%–100.0%) |
| Erased history | probed | linear | 50.7% (45.7%–54.7%) | 50.4% (46.5%–56.2%) |
| Erased history | probed | quadratic | 49.6% (41.8%–55.1%) | 51.5% (48.0%–57.0%) |
| 16-state leaky bank | passive | linear | 100.0% (100.0%–100.0%) | 49.9% (49.6%–50.4%) |
| 16-state leaky bank | passive | quadratic | 100.0% (100.0%–100.0%) | 100.0% (100.0%–100.0%) |
| Linear wave field | incremental | linear | 52.4% (46.1%–57.4%) | 51.5% (48.8%–54.7%) |
| Linear wave field | incremental | quadratic | 50.9% (48.8%–53.9%) | 49.6% (46.1%–52.7%) |
| Linear wave field | passive | linear | 100.0% (100.0%–100.0%) | 50.5% (49.2%–53.5%) |
| Linear wave field | passive | quadratic | 100.0% (100.0%–100.0%) | 96.8% (96.1%–98.0%) |
| Reciprocal field | incremental | linear | 94.8% (87.9%–97.7%) | 88.0% (84.0%–92.6%) |
| Reciprocal field | incremental | quadratic | 98.0% (91.8%–100.0%) | 97.7% (95.7%–99.6%) |
| Reciprocal field | passive | linear | 100.0% (100.0%–100.0%) | 99.5% (98.8%–100.0%) |
| Reciprocal field | passive | quadratic | 100.0% (100.0%–100.0%) | 100.0% (100.0%–100.0%) |
| Shuffled train/validation labels | passive | linear | 61.9% (27.7%–94.1%) | 45.3% (38.3%–56.2%) |
| Shuffled train/validation labels | passive | quadratic | 57.0% (30.5%–82.4%) | 47.1% (2.3%–94.9%) |
| Uncoupled nonlinear cells | incremental | linear | 95.1% (88.7%–97.7%) | 88.3% (83.2%–92.6%) |
| Uncoupled nonlinear cells | incremental | quadratic | 98.0% (91.8%–100.0%) | 97.9% (96.5%–99.6%) |
| Uncoupled nonlinear cells | passive | linear | 100.0% (100.0%–100.0%) | 99.5% (98.8%–100.0%) |
| Uncoupled nonlinear cells | passive | quadratic | 100.0% (100.0%–100.0%) | 100.0% (100.0%–100.0%) |

Reciprocal minus the best alternative, using a shared quadratic reader: **+0.00 percentage points**, relative to **Driven field**. This comparison averages the two targets.

## What was actually tested

- 5 environments, seeds `[7, 19, 31, 43, 59]`.
- Per environment: 256 training, 128 validation, 256 test episodes.
- Two independently varied signed pulses, then a quiet period and a common positive probe.
- Passive measurements: four spatial sensors × two complex quadratures × two reading times = 16 raw readings/features per trajectory.
- Sensor noise standard deviation 0.001. Incremental response subtracts two noisy cloned continuations.
- A single field trajectory and the state-matched delay line each retain 2318 real variables; the leaky bank retains 16.
- Incremental response requires two replicas: 4636 real state variables and 32 raw quadrature readings to produce 16 difference features.
- Linear readers have 16 input features; quadratic readers have 152 monomial features, plus an intercept.
- Feature scaling, PCA, and ridge tuning use training/validation episodes only.
- A separate reader is trained for each environment. Transfer to an unseen environment was not tested.

## Numerical and provenance checks

Maximum adapter/archive error in the five-environment fidelity audit: `9.44e-16`.
The unchanged original program is in `archive/`; extraction hashes are in `archive/PROVENANCE.md`.
The adapted field has a 24 × 24 grid and one pair, with an added external source. Its finite-state equations match the archive.
The archive’s wave/scout timestep mismatch, fixed fractal landscape, and swapped gradient names are retained.

| Environment | Computer | Passive clamps | Probed clamps | Maximum amplitude |
|---|---|---:|---:|---:|
| 7 | Reciprocal field | 0 | 0 | 8.2100 |
| 7 | Driven field | 0 | 0 | 8.1807 |
| 7 | Uncoupled nonlinear cells | 0 | 0 | 8.1909 |
| 7 | Linear wave field | 0 | 0 | 11.2500 |
| 7 | Erased history | 0 | 0 | 1.5444 |
| 19 | Reciprocal field | 0 | 0 | 7.0972 |
| 19 | Driven field | 0 | 0 | 7.0781 |
| 19 | Uncoupled nonlinear cells | 0 | 0 | 7.0892 |
| 19 | Linear wave field | 0 | 0 | 11.0720 |
| 19 | Erased history | 0 | 0 | 1.5097 |
| 31 | Reciprocal field | 0 | 0 | 5.0030 |
| 31 | Driven field | 0 | 0 | 5.0194 |
| 31 | Uncoupled nonlinear cells | 0 | 0 | 5.0305 |
| 31 | Linear wave field | 0 | 0 | 9.2330 |
| 31 | Erased history | 0 | 0 | 1.2665 |
| 43 | Reciprocal field | 0 | 0 | 6.3455 |
| 43 | Driven field | 0 | 0 | 6.3382 |
| 43 | Uncoupled nonlinear cells | 0 | 0 | 6.3492 |
| 43 | Linear wave field | 0 | 0 | 10.7749 |
| 43 | Erased history | 0 | 0 | 1.4769 |
| 59 | Reciprocal field | 0 | 0 | 8.2179 |
| 59 | Driven field | 0 | 0 | 8.2176 |
| 59 | Uncoupled nonlinear cells | 0 | 0 | 8.2280 |
| 59 | Linear wave field | 0 | 0 | 11.3452 |
| 59 | Erased history | 0 | 0 | 1.5422 |

## Boundaries of the result

These are synthetic two-pulse tasks with held-out amplitudes and timings drawn from the same distribution. The four sign combinations occur in training. This establishes task-level computation, not general intelligence, long-context ability, or physical efficiency.

The incremental probe measurement uses identical stochastic continuations of reproducible cloned states. A single physical observation does not provide that counterfactual subtraction.

The state-matched baseline also learns a PCA projection. Retained state and final reader sizes match; total parameter storage and physical energy costs do not. No hardware speed or energy measurement was made.

The leaky and delay/PCA controls receive exact scalar inputs and noiseless state observations. Field observations contain sensor noise. These idealized controls are favored by that asymmetry; the comparison is not a matched noisy-hardware comparison.

The shuffled-label control randomizes both training and validation labels, retaining true test labels. It is one random fit per environment. Its seed outcomes vary widely on this low-dimensional four-history task; it is not a uniformly chance-level null or a calibrated permutation significance test.

The first execution aborted because the linear control reached a shared amplitude cap. The correction removes caps/rescaling from that control to restore linear superposition; the archived nonlinear modes and success thresholds were not tuned. See `EXPERIMENT_LOG.md` and the protocol amendment. This is a documented development experiment, not an independent preregistration.

The fractal landscape remains fixed in every field variant. Its benefit was not separately tested. The archived names “instanton”, “information mass”, and “topological charge” carry no new physical validation here.

## Reproduce

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m fieldcomputer.experiment
MPLBACKEND=Agg python -m fieldcomputer.report
```

The measured run took 322.3 seconds in the working environment. `--smoke` runs a smaller execution check and must not be cited as the frozen experiment.

Complete machine-readable seed results: [`results/results.json`](results/results.json).

Related established work: [Hughes et al., Wave physics as an analog recurrent neural network (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6924985/). That work derives the wave/RNN mapping and trains a simulated wave medium; it does not validate this archived model.
