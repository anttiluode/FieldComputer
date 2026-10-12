# Archived Field Computation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. This experiment was authorized by the user's “well find out” instruction following the proposed input–output probe.

**Goal:** Run and publish a reproducible test of the archived field's memory, nonlinear computation, history-conditioned probe response, and comparative value.

**Architecture:** Preserve the original program and verify a NumPy batch adapter against it. Generate independently reset, balanced pulse episodes, collect sparse observations and fair memory controls, and evaluate validation-selected ridge readers. Publish the protocol, complete seed-level data, report, and tests in the supplied GitHub repository.

**Tech Stack:** Python 3.12+, NumPy, SciPy (archive/fidelity tests), unittest; deterministic JSON and SVG/HTML reports.

**Spec:** `PROTOCOL.md`.

## Global Constraints

- Preserve archive bytes and record SHA-256 provenance.
- Grid 24 × 24, one pair, wave timestep 0.005, scout timestep 0.01.
- Seeds 7, 19, 31, 43, 59; split counts 256/128/256; 16 measured features.
- Observation noise 0.001; no tuning against test results.
- G0–G4 thresholds exactly as written in the protocol.
- Preserve upstream license and deployment workflow.

## Review Focus

- Source forcing and sparse readers must not accidentally receive target labels.
- Probe subtraction must use the same stochastic continuation in both replicas.
- PCA, feature scaling, and ridge selection must never fit on test episodes.
- Linear controls must not acquire nonlinear effects through clipping.
- A negative experiment must still finish successfully and publish its failures.

### Task 1: Dynamics and provenance

**Files:** `archive/scout_com_instanton2.py`, `archive/PROVENANCE.md`, `fieldcomputer/dynamics.py`, `tests/test_dynamics.py`.

**Interfaces:** `FieldBatch(seed, count, mode, grid_size=24)`, `step(source=None, noise=None)`, `clone()`, `observe()`; modes reciprocal/driven/uncoupled/linear.

- [ ] Write fidelity, reciprocal ablation, linear superposition, and reset tests; run them before implementation.
- [ ] Extract the exact archive and implement a batched adapter with forcing as an additive acceleration.
- [ ] Compare several steps and all causal scout states against the literal archive; run the complete dynamics suite.
- [ ] Commit the protocol and verified dynamics.

### Task 2: Held-out experiment

**Files:** `fieldcomputer/experiment.py`, `tests/test_experiment.py`, `requirements.txt`.

**Interfaces:** `make_trials(seed, counts)`, `collect(seed, mode, trials)`, `evaluate(features, targets, splits, family)`, `run_experiment(output)`.

- [ ] Write meaningful tests for balanced disjoint episodes, held-out preprocessing, quadratic parity recovery, state erasure, and gate failure reporting; run before implementation.
- [ ] Implement frozen pulse scheduling, matched counterfactual probes, baseline states, training-only PCA, and validation-selected ridge readers.
- [ ] Run correctness checks, then all five seeds once. Save every seed/model/target score and numerical diagnostic.
- [ ] Commit implementation and measured JSON.

### Task 3: Review and publication

**Files:** `README.md`, `REPORT.md`, `fieldcomputer/report.py`, `index.html`, `results/summary.svg`, `.github/workflows/ci.yml`.

- [ ] Generate the report directly from measured data, distinguishing all four substantive gates.
- [ ] Run the full suite, report consistency checks, and a fresh review; resolve material findings.
- [ ] Commit, push a research branch using GitHub, publish through a pull request, and verify the remote commit and CI.
