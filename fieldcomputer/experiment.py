"""Frozen pulse-memory, delayed-parity, and counterfactual-probe experiment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from .dynamics import FieldBatch

SEEDS = (7, 19, 31, 43, 59)
COUNTS = (256, 128, 256)
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
TASKS = ('first_sign', 'parity')
NOISE = .001


def make_trials(seed, counts=COUNTS):
    if len(counts) != 3 or any(n < 4 or n % 4 for n in counts):
        raise ValueError('Three positive split counts divisible by four are required')
    source, targets, splits, descriptions = [], [], [], []
    offset = 0
    for split, count in enumerate(counts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 100, split]))
        signs = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (count//4, 1))
        rng.shuffle(signs)
        amplitude = rng.uniform(.6, 1.4, (count, 2))
        start = np.column_stack((rng.integers(8, 17, count), rng.integers(112, 129, count)))
        duration = rng.integers(48, 73, (count, 2))
        signal = np.zeros((count, 336))
        clock = np.arange(336)[None, :]
        for pulse in range(2):
            active = (clock >= start[:, pulse, None]) & (clock < (start[:, pulse]+duration[:, pulse])[:, None])
            signal += active*(4.*amplitude[:, pulse]*signs[:, pulse])[:, None]
        source.append(signal)
        targets.append(np.column_stack((signs[:, 0], signs[:, 0]*signs[:, 1])))
        splits.append(np.arange(offset, offset+count))
        descriptions.append(np.column_stack((signs, amplitude, start, duration)))
        offset += count
    return {'source': np.concatenate(source), 'targets': np.concatenate(targets),
            'splits': tuple(splits), 'description': np.concatenate(descriptions)}


def collect(seed, mode, trials, erase=False):
    inputs = trials['source']
    count = len(inputs)
    model = FieldBatch(seed, count, mode)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    for t in range(inputs.shape[1]):
        model.step(inputs[:, t], .02*noise_rng.standard_normal((count, 2)))
    if erase:
        model.reset()
    passive, probed = model.clone(), model.clone()
    observation_rng = np.random.default_rng(np.random.SeedSequence([seed, 300]))
    passive_reads, probed_reads = [], []
    # Probe at relative steps 0..47; readings after 20 and 60 quiet steps.
    for t in range(108):
        noise = .02*noise_rng.standard_normal((count, 2))
        passive.step(np.zeros(count), noise)
        probed.step(np.full(count, 4. if t < 48 else 0.), noise)
        if t in (67, 107):
            passive_reads.append(passive.observe()+NOISE*observation_rng.standard_normal((count, 8)))
            probed_reads.append(probed.observe()+NOISE*observation_rng.standard_normal((count, 8)))
    quiet = np.concatenate(passive_reads, axis=1)
    actual = np.concatenate(probed_reads, axis=1)
    features = {'passive': quiet, 'incremental': actual-quiet, 'probed': actual}
    diagnostics = {'clamps_passive': passive.clamp_count, 'clamps_probed': probed.clamp_count,
        'energy_rescales': max(passive.energy_rescales, probed.energy_rescales),
        'max_abs': max(passive.max_abs, probed.max_abs),
        'finite': all(np.isfinite(x).all() for x in features.values()),
        'real_state_size': model.real_state_size, 'observed_features': 16,
        'incremental_rms': float(np.sqrt(np.mean((actual-quiet)**2)))}
    if mode == 'linear' and (diagnostics['clamps_passive'] or diagnostics['clamps_probed'] or diagnostics['energy_rescales']):
        raise RuntimeError('Linear control clipped or rescaled; the comparison is invalid')
    return features, diagnostics


def training_pca(states, train_ids, feature_count=16):
    mean = states[train_ids].mean(axis=0)
    centered = states-mean
    train = centered[train_ids]
    # Dual PCA avoids an unnecessary decomposition of the large state matrix.
    values, vectors = np.linalg.eigh(train @ train.T)
    order = np.argsort(values)[::-1]
    keep = order[values[order] > max(1e-12, values[order[0]]*1e-12)][:feature_count]
    basis = np.zeros((feature_count, states.shape[1]))
    if len(keep):
        basis[:len(keep)] = (vectors[:, keep].T @ train) / np.sqrt(values[keep])[:, None]
    return centered @ basis.T, {'mean': mean, 'basis': basis}


def baseline_features(trials, state_size):
    history = np.pad(trials['source'], ((0, 0), (0, 108)))
    # Retain exactly state_size samples; omitted older inputs are initial zeros.
    state = np.pad(history[:, -state_size:], ((0, 0), (max(0, state_size-history.shape[1]), 0)))
    delay, _ = training_pca(state, trials['splits'][0])
    decay = np.exp(-1/np.geomspace(4., 1024., 16))
    leaky = np.zeros((len(history), 16))
    for t in range(history.shape[1]):
        leaky = decay*leaky+(1-decay)*history[:, t, None]
    return {'leaky': leaky, 'delay_pca': delay}


def _standardize(values, ids):
    mean = values[ids].mean(axis=0)
    std = values[ids].std(axis=0)
    return (values-mean)/np.where(std > 1e-12, std, 1.)


def evaluate(features, targets, splits, family):
    if family not in ('linear', 'quadratic') or features.shape[1] != 16:
        raise ValueError('Readers require 16 features and a known reader family')
    train, validation, test = splits
    design = _standardize(features, train)
    if family == 'quadratic':
        i, j = np.triu_indices(16)
        design = np.concatenate((design, design[:, i]*design[:, j]), axis=1)
        design = _standardize(design, train)
    x = design[train]
    y_mean = targets[train].mean(axis=0)
    y = targets[train]-y_mean
    gram = x.T @ x / len(train)
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 0.)
    cross = vectors.T @ (x.T @ y / len(train))
    candidates = []
    for alpha in ALPHAS:
        weights = vectors @ (cross/(values+alpha)[:, None])
        prediction = design @ weights + y_mean
        accuracy = np.mean(np.where(prediction[validation] >= 0, 1., -1.) == targets[validation], axis=0)
        mse = np.mean((prediction[validation]-targets[validation])**2, axis=0)
        candidates.append((alpha, prediction, accuracy, mse))
    predictions = np.zeros((len(test), 2))
    tasks, alphas = {}, []
    for column, name in enumerate(TASKS):
        best = max(candidates, key=lambda c: (c[2][column], -c[3][column], -c[0]))
        alpha, prediction, accuracy, _ = best
        pred = prediction[test, column]
        correct = np.where(pred >= 0, 1., -1.) == targets[test, column]
        predictions[:, column] = pred
        alphas.append(alpha)
        tasks[name] = {'accuracy': float(correct.mean()), 'correct': int(correct.sum()),
            'test_count': len(test), 'validation_accuracy': float(accuracy[column]),
            'mse': float(np.mean((pred-targets[test, column])**2)), 'alpha': alpha}
    return {'tasks': tasks, 'alphas': alphas, 'predictions': predictions,
            'reader_features': design.shape[1]}


def compute_gates(means, memory_seeds, advantage_seeds, fidelity):
    memory = means[('reciprocal', 'passive', 'linear')]['first_sign']
    parity = means[('reciprocal', 'passive', 'linear')]['parity']
    linear_parity = means[('linear', 'passive', 'linear')]['parity']
    response = means[('reciprocal', 'incremental', 'linear')]['parity']
    linear_response = means[('linear', 'incremental', 'linear')]['parity']
    score = lambda model: np.mean(list(means[(model, 'passive', 'quadratic')].values()))
    alternatives = {model: float(score(model)) for model in ('driven', 'uncoupled', 'linear', 'leaky', 'delay_pca')}
    best = max(alternatives, key=alternatives.get)
    advantage = float(score('reciprocal')-alternatives[best])
    return {
        'G0': {'pass': bool(fidelity), 'meaning': 'Numerical archive fidelity and finite runs'},
        'G1': {'pass': bool(memory >= .80 and min(memory_seeds) >= .75), 'accuracy': memory,
            'min_seed_accuracy': min(memory_seeds), 'meaning': 'Readable first-pulse memory'},
        'G2': {'pass': bool(parity >= .80 and parity-linear_parity >= .10), 'accuracy': parity,
            'linear_control_accuracy': linear_parity, 'meaning': 'Nonlinear delayed parity'},
        'G3': {'pass': bool(response >= .80 and response-linear_response >= .10), 'accuracy': response,
            'linear_control_accuracy': linear_response, 'meaning': 'History-conditioned incremental probe response'},
        'G4': {'pass': bool(advantage >= .05 and sum(d > 0 for d in advantage_seeds) >= 4),
            'difference': advantage, 'best_alternative': best, 'alternative_scores': alternatives,
            'positive_seed_count': sum(d > 0 for d in advantage_seeds),
            'meaning': 'Advantage over alternative computers with the same reader family'}
    }


def fidelity_check():
    path = Path(__file__).resolve().parents[1]/'archive/scout_com_instanton2.py'
    spec = importlib.util.spec_from_file_location('fieldcomputer_literal_archive', path)
    archive = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = archive
    spec.loader.exec_module(archive)
    worst = 0.
    for seed in SEEDS:
        original = archive.MultiScaleFieldComputer(grid_size=24)
        np.random.seed(seed)
        original.create_instanton_scout_pair((8, 12), amplitude=1.2)
        model = FieldBatch(seed, 1, 'reciprocal')
        for _ in range(20):
            state = np.random.get_state()
            noise = .02*np.random.randn(1, 2)
            np.random.set_state(state)
            original.step()
            model.step(noise=noise)
            worst = max(worst, float(np.max(np.abs(model.phi[0]-original.phi))),
                float(np.max(np.abs(model.scout_pos[0]-original.pairs[0].scout_pos))),
                float(np.max(np.abs(model.scout_vel[0]-original.pairs[0].scout_vel))))
    return {'pass': worst <= 1e-10, 'max_abs_error': worst,
            'environments': len(SEEDS), 'unforced_steps_per_environment': 20}


def run_experiment(output, smoke=False):
    root = Path(__file__).resolve().parents[1]
    start = time.perf_counter()
    seeds, counts = ((SEEDS[0],), (32, 16, 32)) if smoke else (SEEDS, COUNTS)
    fidelity = fidelity_check()
    if not fidelity['pass']:
        raise RuntimeError('Archive fidelity failed')
    rows, diagnostics, memory_seeds, advantage_seeds = [], [], [], []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        trials = make_trials(seed, counts)
        saved = {'source': trials['source'], 'targets': trials['targets']}
        seed_scores = {}

        def score(model, observation, features, labels=None, state_size=None):
            for family in ('linear', 'quadratic'):
                result = evaluate(features, trials['targets'] if labels is None else labels, trials['splits'], family)
                key = (model, observation, family)
                seed_scores[key] = {task: result['tasks'][task]['accuracy'] for task in TASKS}
                rows.append({'seed': seed, 'model': model, 'observation': observation,
                    'family': family, 'tasks': result['tasks'], 'observed_features': 16,
                    'reader_features': result['reader_features'], 'retained_real_state': state_size})
            saved[f'{model}_{observation}'] = features

        for mode in FieldBatch.MODES:
            features, diag = collect(seed, mode, trials)
            diagnostics.append({'seed': seed, 'model': mode, **diag})
            for observation in ('passive', 'incremental'):
                score(mode, observation, features[observation], state_size=diag['real_state_size'])
            if mode == 'reciprocal':
                reciprocal_passive = features['passive']
            print(f'Seed {seed}, {mode}: memory {seed_scores[(mode, "passive", "linear")]["first_sign"]:.3f}, '
                  f'parity {seed_scores[(mode, "passive", "linear")]["parity"]:.3f}, '
                  f'probe parity {seed_scores[(mode, "incremental", "linear")]["parity"]:.3f}', flush=True)
        state_size = FieldBatch(seed, 1, 'reciprocal').real_state_size
        for model, features in baseline_features(trials, state_size).items():
            score(model, 'passive', features, state_size=16 if model == 'leaky' else state_size)
        erased, diag = collect(seed, 'reciprocal', trials, erase=True)
        diagnostics.append({'seed': seed, 'model': 'erased', **diag})
        score('erased', 'probed', erased['probed'], state_size=state_size)
        shuffled = trials['targets'].copy()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 400]))
        for ids in trials['splits'][:2]:
            shuffled[ids] = shuffled[rng.permutation(ids)]
        score('shuffled_labels', 'passive', reciprocal_passive, shuffled, state_size)
        memory_seeds.append(seed_scores[('reciprocal', 'passive', 'linear')]['first_sign'])
        qscore = lambda model: np.mean(list(seed_scores[(model, 'passive', 'quadratic')].values()))
        advantage_seeds.append(float(qscore('reciprocal')-max(qscore(m) for m in ('driven', 'uncoupled', 'linear', 'leaky', 'delay_pca'))))
        np.savez_compressed(output.parent/f'features_seed_{seed}.npz', **saved)
        print(f'Seed {seed} complete; quadratic advantage {advantage_seeds[-1]:+.3f}', flush=True)
    keys = sorted({(row['model'], row['observation'], row['family']) for row in rows})
    means, summaries = {}, []
    for key in keys:
        selected = [row for row in rows if (row['model'], row['observation'], row['family']) == key]
        metrics = {task: {'mean': float(np.mean([r['tasks'][task]['accuracy'] for r in selected])),
            'min': min(r['tasks'][task]['accuracy'] for r in selected),
            'max': max(r['tasks'][task]['accuracy'] for r in selected)} for task in TASKS}
        means[key] = {task: metrics[task]['mean'] for task in TASKS}
        summaries.append({'model': key[0], 'observation': key[1], 'family': key[2], 'tasks': metrics})
    gates = compute_gates(means, memory_seeds, advantage_seeds,
        fidelity['pass'] and all(d['finite'] for d in diagnostics))
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root/'PROTOCOL.md', root/'archive/scout_com_instanton2.py',
                     root/'fieldcomputer/dynamics.py', root/'fieldcomputer/experiment.py')}
    result = {'schema_version': 1, 'run_kind': 'smoke' if smoke else 'frozen_full',
        'seeds': list(seeds), 'counts_per_seed': dict(zip(('train', 'validation', 'test'), counts)),
        'numpy_version': np.__version__, 'seconds': time.perf_counter()-start,
        'source_sha256': source_hashes, 'fidelity': fidelity, 'gates': gates,
        'diagnostics': diagnostics, 'seed_results': rows, 'summary': summaries,
        'quadratic_seed_differences_vs_best_alternative': advantage_seeds,
        'measurement_noise_std': NOISE, 'field_real_state': 2318,
        'reader_budget': {'observations_or_projected_features': 16, 'linear': 16, 'quadratic': 152}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(gates, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/results.json')
    parser.add_argument('--smoke', action='store_true', help='Small execution check; not a research result')
    args = parser.parse_args()
    run_experiment(args.output, args.smoke)


if __name__ == '__main__':
    main()
