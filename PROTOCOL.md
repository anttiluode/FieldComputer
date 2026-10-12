# FieldComputer: frozen experiment, 12 October 2026

## Question and scope

Does the archived 2025 `MultiScaleFieldComputer` support readable temporal
computation, and do its nonlinear spatial dynamics and reciprocal scout coupling
contribute to performance? This is a numerical experiment, not a physical
hardware demonstration or a claim about cognition.

The archive is preserved verbatim. A batched adapter retains its wave update,
amplitude limits, wave/scout forces, center tracking, and adaptive coupling.
It uses a 24 × 24 grid, one scout/wave pair, and local random generators.
An externally specified Gaussian source is added to the wave acceleration.
The effective wave timestep remains 0.005; the archive's scout timestep remains
0.01. That mismatch is preserved and disclosed, not silently repaired.

## Inputs, observations, and targets

Each independently reset episode receives two signed pulses, a quiet interval,
and the same positive probe. The pulse signs are balanced over all four
combinations. Amplitudes and durations vary independently of the signs.
At the time of observation the external input is zero in every episode.
Targets are (1) the sign of the first pulse and (2) the product of the two
signs (delayed XOR/parity). The reader never receives the signs or input
history for field representations.

Read four fixed spatial sensors, two complex quadratures, at two fixed times:
16 features. Fit ridge readers on training episodes; choose regularization
using validation episodes; evaluate once on held-out test episodes.
Report both linear and quadratic readers. Only continuous amplitudes/timings
are held out; pulse-sign combinations are represented during training.
Observation noise is 0.001 per quadrature, fixed in physical field units.

Exact schedule: first pulse starts at integer step 8–16 and lasts 48–72
steps; second starts at 112–128 and lasts 48–72 steps. Each amplitude is
uniform on [0.6, 1.4], multiplied by its sign and source strength 4. Source
center is (13,12), Gaussian width 2. Histories end at step 336; the positive
probe has strength 4 for 48 steps. Observations follow another 20 and 60
quiet steps. Sensors are (11,10), (14,10), (11,14), (14,14). The pair starts
at (8,12) with amplitude 1.2 and radius 8. Initial phase and scout offset
vary by environment seed. The fractal landscape retains archive seed 42.
Ridge regularization grid: 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100.
Leaky-bank time constants are 16 logarithmically spaced values from 4 to
1024 steps. PCA fits training states only. Quadratic readers use all
degree-two monomials of the same 16 measured/projected features.

Also measure the incremental response to the probe by subtracting a cloned
continuation without the probe. This counterfactual measurement isolates a
change in response from a residual wave left by the earlier pulses. It requires
reproducible replicas in simulation, rather than a single physical observation.

## Controls

1. Reciprocal nonlinear field: the archived equations, plus the external source.
2. Driven nonlinear field: identical, but scout feedback into the field is zero.
3. Uncoupled nonlinear cells: driven field with its spatial Laplacian also zero.
4. Linear field: constant wave speed, linear potential, no collision/cubic
   potential, no scout feedback. Abort a claimed linear comparison if it clips.
5. A 16-state leaky bank driven by the same scalar signal.
6. A delay line with the same number of stored real state variables as the
   reciprocal simulator, reduced to 16 features by training-only PCA.
7. An erased-history field, reset before the common probe.
8. Shuffled training labels, with original held-out labels for evaluation.

All learned readers use the same feature count, ridge grid, splits, and reader
families. The delay-line control matches total retained numerical state;
the leaky bank is much smaller. Field-only controls share the same wave grid.
Raw-history controls are allowed access to past input because they are
alternative computers, not readers of the field.

## Frozen sample counts and gates

Environment seeds: 7, 19, 31, 43, 59. Each seed has 256 training, 128 validation,
and 256 test episodes, generated independently. No parameter search after
viewing the test outcomes. Chance is 50% for both balanced binary targets.

- G0, fidelity: the adapter agrees with the literal archive on unforced steps
  to absolute tolerance 1e-10, and every run remains finite. Record clamps.
- G1, readable memory: reciprocal passive measurements, linear reader, mean
  first-sign accuracy ≥ 80%, and ≥ 75% on every environment seed.
- G2, nonlinear computation: reciprocal passive measurements, linear reader,
  mean parity accuracy ≥ 80%, and exceeds the linear-field parity result
  by at least 10 percentage points.
- G3, changed response: reciprocal incremental-probe measurements, linear
  reader, mean parity accuracy ≥ 80%, and exceeds the linear incremental
  control by at least 10 percentage points.
- G4, advantage: using the same quadratic reader for every representation,
  reciprocal mean accuracy across both targets exceeds the best alternative
  (driven field, uncoupled cells, linear field, leaky bank, matched delay line)
  by ≥ 5 percentage points overall, with a positive difference on ≥ 4/5 seeds.

G1/G2 establish computation only on these synthetic tasks. G3 tests a
history-conditioned response using cloned simulations. G4 is the stronger
architectural claim and can fail even if the others pass. Seed results and
all controls are published, including failures. Report observation/state
budgets, feature preprocessing, reset semantics, and provenance.

## Corrective amendment after the first aborted execution

The first execution stopped at the linear-control guard: removing the cubic
potential allowed some linear trajectories to reach the archive's shared
amplitude cap. A capped wave is not a linear control. The linear control now
omits both amplitude capping and energy rescaling and still aborts on nonfinite
values. Its positive linear potential is retained. A full-schedule superposition
regression verifies that the probe increment is independent of history even
when amplitude exceeds 10. The original nonlinear modes, inputs, splits,
readers, and gate thresholds were not adjusted. This correction was made after
seeing the first environment's provisional nonlinear results; this is a
documented development protocol, not an independently preregistered study.
