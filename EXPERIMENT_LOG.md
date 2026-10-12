# Experiment execution record

Plan: `docs/superpowers/plans/2026-10-12-field-experiment.md`.

- Pre-flight: dynamics produces a 24 × 24 complex current/previous state and
  8 sensor quadratures per observation. Experiment consumes two observations
  as 16 features. Reports consume measured JSON. Interfaces agree.
- Task 1: literal archive preserved; four numerical/causal checks passed
  after first failing for the missing adapter. Commit `dc4eb8f`.
- Task 2: balanced episodes, training-only PCA, quadratic parity recovery,
  test-label isolation, and failed-advantage reporting checks passed after
  first failing for missing functions. Protocol implementation committed
  as `289a84d` before collecting task results.
- Ruling: implement inline in a fresh research checkout/branch and publish
  in the repository supplied by the user. The request “well find out” is
  authorization for the experiment proposed immediately beforehand.
- First full execution: aborted during seed 7's linear control because the
  shared archive cap made that control nonlinear. No completed result file
  was produced. The seed 7 nonlinear provisional metrics were visible before
  the correction.
- Ruling: remove amplitude capping and energy rescaling from the linear
  control, retaining its constant-speed/linear-potential equations. This
  restores superposition without changing the archived nonlinear models,
  task distribution, reader budget, or success thresholds. If wrong, the
  linear-control comparison would be invalid; the new full-schedule
  superposition regression directly checks that boundary.
- Corrected full execution: completed all five seeds in 322.3 seconds.
  G0–G3 passed; G4 failed with zero comparative improvement. All nonlinear
  and linear runs remained finite with zero clamps/rescalings.
- Fresh review: reviewer independently ran all 10 tests and reproduced all
  110 non-shuffled reader evaluations from saved features, including the
  validation-selected regularization values. Source hashes matched.
- Final reporting fix: the first report draft conflated per-trajectory
  budgets with cloned incremental-probe budgets. The report now explicitly
  gives 32 raw measurements and 4636 real state variables for the latter;
  a report regression failed before the correction and passed afterward.
- Report clarifications: the shuffled-label control shuffles both training
  and validation labels; its scores vary widely across seeds. Input-history
  baselines use noiseless observations. The driven field and uncoupled cells
  match reciprocal passive scores. No separate spatial/fractal benefit or
  physical efficiency has been shown.
