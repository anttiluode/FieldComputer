"""Generate Markdown, SVG, and a static research report from measured JSON."""
import html
import json
from pathlib import Path

NAMES = {'reciprocal': 'Reciprocal field', 'driven': 'Driven field',
    'uncoupled': 'Uncoupled nonlinear cells', 'linear': 'Linear wave field',
    'leaky': '16-state leaky bank', 'delay_pca': 'State-matched delay line + PCA',
    'erased': 'Erased history', 'shuffled_labels': 'Shuffled train/validation labels'}


def interpretation(data):
    gates = data['gates']
    if gates['G1']['pass'] and gates['G2']['pass']:
        lead = 'This tested adaptation of the archived field retained readable history and performed nonlinear delayed-parity classification.'
    elif gates['G1']['pass']:
        lead = 'This tested adaptation retained readable history; nonlinear delayed-parity computation did not pass its gate.'
    else:
        lead = 'The field did not pass the frozen readable-memory gate on this task.'
    if gates['G3']['pass']:
        lead += ' Its incremental response to a common probe also carried history information.'
    else:
        lead += ' The incremental probe response did not pass the history-conditioned-response gate.'
    if gates['G4']['pass']:
        lead += ' The comparative-advantage gate passed on these tasks.'
    else:
        lead += ' No comparative advantage for reciprocal coupling was established.'
    lookup = {(r['model'], r['observation'], r['family']): r for r in data['summary']}
    reciprocal = lookup.get(('reciprocal', 'passive', 'linear'))
    controls = [lookup.get((m, 'passive', 'linear')) for m in ('driven', 'uncoupled')]
    if reciprocal and all(controls) and all(abs(r['tasks'][t]['mean']-reciprocal['tasks'][t]['mean']) < 1e-12 for r in controls for t in ('first_sign', 'parity')):
        lead += ' The driven field and uncoupled nonlinear cells matched its passive scores; spatial interactions and scout feedback were unnecessary on these tasks.'
    return lead


def render_report(data):
    counts = data['counts_per_seed']
    gates = data['gates']
    lines = ['# FieldComputer: measured result', '', interpretation(data), '',
        '## Gates', '', '| Gate | Result | Claim tested |', '|---|---|---|']
    for name, gate in gates.items():
        lines.append(f'| {name} | {"PASS" if gate["pass"] else "FAIL"} | {gate["meaning"]} |')
    lines += ['', '## Held-out accuracy', '',
        'Means across environments; ranges are the minimum and maximum seed accuracy. Chance is 50%.', '',
        '| Computer | Observation | Reader | First-pulse sign | Delayed parity |', '|---|---|---|---:|---:|']
    for row in data['summary']:
        def metric(task):
            m = row['tasks'][task]
            return f'{m["mean"]:.1%} ({m["min"]:.1%}–{m["max"]:.1%})'
        lines.append(f'| {NAMES[row["model"]]} | {row["observation"]} | {row["family"]} | {metric("first_sign")} | {metric("parity")} |')
    advantage = gates['G4']
    lines += ['', 'Reciprocal minus the best alternative, using a shared quadratic reader: '
        f'**{advantage["difference"]*100:+.2f} percentage points**, relative to '
        f'**{NAMES[advantage["best_alternative"]]}**. This comparison averages the two targets.', '',
        '## What was actually tested', '',
        f'- {len(data["seeds"])} environments, seeds `{data["seeds"]}`.',
        f'- Per environment: {counts["train"]} training, {counts["validation"]} validation, {counts["test"]} test episodes.',
        '- Two independently varied signed pulses, then a quiet period and a common positive probe.',
        '- Passive measurements: four spatial sensors × two complex quadratures × two reading times = 16 raw readings/features per trajectory.',
        '- Sensor noise standard deviation 0.001. Incremental response subtracts two noisy cloned continuations.',
        '- A single field trajectory and the state-matched delay line each retain 2318 real variables; the leaky bank retains 16.',
        '- Incremental response requires two replicas: 4636 real state variables and 32 raw quadrature readings to produce 16 difference features.',
        '- Linear readers have 16 input features; quadratic readers have 152 monomial features, plus an intercept.',
        '- Feature scaling, PCA, and ridge tuning use training/validation episodes only.',
        '- A separate reader is trained for each environment. Transfer to an unseen environment was not tested.', '',
        '## Numerical and provenance checks', '',
        f'Maximum adapter/archive error in the five-environment fidelity audit: `{data["fidelity"]["max_abs_error"]:.3g}`.',
        'The unchanged original program is in `archive/`; extraction hashes are in `archive/PROVENANCE.md`.',
        'The adapted field has a 24 × 24 grid and one pair, with an added external source. Its finite-state equations match the archive.',
        'The archive’s wave/scout timestep mismatch, fixed fractal landscape, and swapped gradient names are retained.', '',
        '| Environment | Computer | Passive clamps | Probed clamps | Maximum amplitude |', '|---|---|---:|---:|---:|']
    for diag in data['diagnostics']:
        lines.append(f'| {diag["seed"]} | {NAMES[diag["model"]]} | {diag["clamps_passive"]} | {diag["clamps_probed"]} | {diag["max_abs"]:.4f} |')
    lines += ['', '## Boundaries of the result', '',
        'These are synthetic two-pulse tasks with held-out amplitudes and timings drawn from the same distribution. '
        'The four sign combinations occur in training. This establishes task-level computation, not general intelligence, long-context ability, or physical efficiency.', '',
        'The incremental probe measurement uses identical stochastic continuations of reproducible cloned states. '
        'A single physical observation does not provide that counterfactual subtraction.', '',
        'The state-matched baseline also learns a PCA projection. Retained state and final reader sizes match; '
        'total parameter storage and physical energy costs do not. No hardware speed or energy measurement was made.', '',
        'The leaky and delay/PCA controls receive exact scalar inputs and noiseless state observations. '
        'Field observations contain sensor noise. These idealized controls are favored by that asymmetry; '
        'the comparison is not a matched noisy-hardware comparison.', '',
        'The shuffled-label control randomizes both training and validation labels, retaining true test labels. '
        'It is one random fit per environment. Its seed outcomes vary widely on this low-dimensional four-history task; '
        'it is not a uniformly chance-level null or a calibrated permutation significance test.', '',
        'The first execution aborted because the linear control reached a shared amplitude cap. '
        'The correction removes caps/rescaling from that control to restore linear superposition; '
        'the archived nonlinear modes and success thresholds were not tuned. '
        'See `EXPERIMENT_LOG.md` and the protocol amendment. This is a documented development experiment, not an independent preregistration.', '',
        'The fractal landscape remains fixed in every field variant. Its benefit was not separately tested. '
        'The archived names “instanton”, “information mass”, and “topological charge” carry no new physical validation here.', '',
        '## Reproduce', '', '```bash', 'python -m pip install -r requirements.txt',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest discover -s tests -v',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m fieldcomputer.experiment',
        'MPLBACKEND=Agg python -m fieldcomputer.report', '```', '',
        f'The measured run took {data["seconds"]:.1f} seconds in the working environment. '
        '`--smoke` runs a smaller execution check and must not be cited as the frozen experiment.', '',
        'Complete machine-readable seed results: [`results/results.json`](results/results.json).', '',
        'Related established work: [Hughes et al., Wave physics as an analog recurrent neural network (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6924985/). '
        'That work derives the wave/RNN mapping and trains a simulated wave medium; it does not validate this archived model.', '']
    return '\n'.join(lines)


def plot_summary(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    models = ('reciprocal', 'driven', 'uncoupled', 'linear', 'leaky', 'delay_pca')
    labels = ['Reciprocal\nfield', 'Driven\nfield', 'Uncoupled\ncells', 'Linear\nfield', 'Leaky\nbank', 'Matched\ndelay line']
    lookup = {(r['model'], r['observation'], r['family']): r for r in data['summary']}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, family in zip(axes, ('linear', 'quadratic')):
        x = np.arange(len(models))
        for offset, task, color, name in ((-.18, 'first_sign', '#147d86', 'First-pulse sign'), (.18, 'parity', '#d18437', 'Delayed parity')):
            values = [lookup[(m, 'passive', family)]['tasks'][task]['mean'] for m in models]
            ax.bar(x+offset, values, .34, color=color, label=name)
            for xp, value in zip(x+offset, values):
                ax.text(xp, value+.018, f'{value:.1%}', ha='center', fontsize=8)
        ax.axhline(.5, color='#747b80', ls='--', lw=1, label='Chance')
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_xticks(x, labels)
        ax.set_ylabel('Held-out accuracy')
        ax.set_title(f'{family.capitalize()} reader · 16 observations/projected features')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.12)
    axes[0].legend(loc='upper left', ncol=3, frameon=False)
    fig.suptitle('FieldComputer · passive memory and nonlinear computation', fontsize=14)
    fig.savefig(path)
    plt.close(fig)
    if Path(path).suffix.lower() == '.svg':
        svg = Path(path).read_text()
        Path(path).write_text('\n'.join(line.rstrip() for line in svg.splitlines())+'\n')


def render_page(data):
    gate_cards = ''.join(f'<article class="gate"><span>{html.escape(name)}</span><strong class="{ "pass" if g["pass"] else "fail"}">{"PASS" if g["pass"] else "FAIL"}</strong><p>{html.escape(g["meaning"])}</p></article>' for name, g in data['gates'].items())
    rows = []
    for r in data['summary']:
        tasks = r['tasks']
        rows.append(f'<tr><td>{NAMES[r["model"]]}</td><td>{r["observation"]}</td><td>{r["family"]}</td><td>{tasks["first_sign"]["mean"]:.1%}</td><td>{tasks["parity"]["mean"]:.1%}</td></tr>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FieldComputer — measured results</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f6f4ee;color:#182b30;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:48px 24px}h1{font-size:clamp(32px,5vw,60px);line-height:1.1;max-width:850px;letter-spacing:-.04em}h2{margin-top:40px}.eyebrow{color:#54757c;letter-spacing:.12em;font-size:12px;text-transform:uppercase}.lead{font-size:20px;max-width:850px}.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:32px 0}.gate{background:white;border:1px solid #d9e0dd;border-radius:12px;padding:18px}.gate span{display:block;font-size:12px}.gate strong{font-size:24px}.gate p{font-size:13px;margin-bottom:0}.pass{color:#147d86}.fail{color:#a46624}.chart{display:block;background:white;width:100%;border-radius:12px;padding:12px}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e0dd}a{color:#096875}code{background:#e9ece7;padding:3px 6px;border-radius:4px}footer{border-top:1px solid #d9e0dd;margin-top:40px;padding-top:20px;font-size:14px}.note{background:#eaf0ec;border-radius:12px;padding:22px}
    </style></head><body><main><p class="eyebrow">2025 scout archive · experiment of 12 October 2026</p><h1>Can the old field compute?</h1>'''+f'<p class="lead">{html.escape(interpretation(data))}</p><p>Five environments. 1,280 held-out episodes. Four spatial sensors. All controls and failures included.</p><div class="gates">{gate_cards}</div>'+'''
    <img class="chart" src="results/summary.svg" alt="Held-out accuracy of the reciprocal field and five alternative computers, with linear and quadratic readers.">
    <h2>Read the past from a later response</h2><p>Two signed pulses enter the wave field. After a quiet interval, the reader tries to recover the first sign and the relationship between the two signs. A cloned continuation with a common probe tests whether history changes the incremental response. Passive readings use one 2318-variable trajectory and 16 raw measurements; the incremental test uses two replicas and 32 raw measurements to produce 16 differences.</p>
    <div class="note"><strong>What this establishes</strong><p>This is a numerical task demonstration in an adaptation of the archived equations. The delay-line baseline matches retained numerical state; the leaky bank uses much less. Those input-history controls have noiseless observations, favoring them in the comparison. The result does not establish physical efficiency, intelligence, or a benefit from the fixed fractal landscape.</p></div>
    <h2>Every measured comparison</h2><p>Mean test accuracy across seeds; chance is 50%. “Incremental” is probed minus unprobed continuation.</p><div class="table"><table><thead><tr><th>Computer</th><th>Observation</th><th>Reader</th><th>First sign</th><th>Parity</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
    <h2>Reproduce and inspect</h2><p><a href="REPORT.md">Full report</a> · <a href="PROTOCOL.md">Protocol and correction</a> · <a href="results/results.json">Complete seed data</a> · <a href="https://github.com/anttiluode/FieldComputer">Source repository</a></p><p><code>python -m fieldcomputer.experiment</code></p><footer>The original 2025 program is preserved with extraction hashes. Numerical agreement is tested against that program. The first execution exposed an invalid capped linear control; the documented correction restores linear superposition. No archived nonlinear parameters or gate thresholds were tuned to improve the task results.</footer></main></body></html>'''


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'results/results.json').read_text())
    if data['run_kind'] != 'frozen_full':
        raise ValueError('A smoke run cannot be published as the full experiment')
    (root/'REPORT.md').write_text(render_report(data))
    (root/'index.html').write_text(render_page(data))
    plot_summary(data, root/'results/summary.svg')
    (root/'README.md').write_text('# FieldComputer\n\n'+interpretation(data)+'\n\n'
        'A reproducible numerical investigation of Antti’s archived scout/field program, with sparse readers and matched controls.\n\n'
        '![Held-out accuracy](results/summary.svg)\n\n'
        '- [Measured report](REPORT.md) — all gates, seed ranges, and limits.\n'
        '- [Frozen protocol and correction](PROTOCOL.md) — inputs, controls, thresholds.\n'
        '- [Complete results](results/results.json) — every seed and reader.\n'
        '- [Original archive](archive/scout_com_instanton2.py) and [provenance](archive/PROVENANCE.md).\n'
        '- [Execution record](EXPERIMENT_LOG.md).\n\n'
        '```bash\npython -m pip install -r requirements.txt\n'
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest discover -s tests -v\n'
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m fieldcomputer.experiment\n'
        'MPLBACKEND=Agg python -m fieldcomputer.report\n```\n\n'
        'The original GUI can be run with `python archive/scout_com_instanton2.py`. '
        '`index.html` displays the measured report through the existing Pages workflow. '
        'A smaller `--smoke` execution check is available; its outputs are not research results.\n\n'
        'The strongest claim evaluated here is comparative value. Passing memory or parity gates alone does not establish that advantage.\n')
    print('Generated REPORT.md, README.md, index.html, results/summary.svg from measured JSON.')


if __name__ == '__main__':
    main()
