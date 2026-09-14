"""Synthetic CPU checks for R22 replay plus verified short-region continuation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import short_region_objective as short
from training import train_unified_short_regions as trainer
from retention_paired_known_sampler import KNOWN_SUPPLEMENT_MARKER
from retention_supplement_sampler import SUPPLEMENT_MARKER


def short_rows(all_lengths=False):
    rows = []
    lengths = range(1, 5) if all_lengths else (1,)
    views = short.SHORT_SAMPLING['views'] if all_lengths else ('native',)
    for target, family in enumerate(short.FAMILIES[:-1]):
        for length in lengths:
            for view in views:
                rows.append({'split': 'train', 'native_font_verified': True, 'target': target,
                             'family': family, 'source_font_family': family, 'glyph_count': length,
                             'domain': 'ios' if target % 2 else 'android',
                             'font_face': f'{family}-face', 'view': view})
    for source in short.SHORT_UNKNOWN_SOURCES:
        for length in lengths:
            for view in views:
                rows.append({'split': 'train', 'native_font_verified': True, 'target': 24,
                             'family': '__unknown__', 'source_font_family': source,
                             'glyph_count': length, 'domain': 'ios',
                             'font_face': f'{source}-face', 'view': view})
    return rows


def loss_rows():
    targets = list(range(24)) + [24] * 8
    rows = [{'split': 'train', 'native_font_verified': True, 'target': target,
             'family': short.FAMILIES[target], 'glyph_count': 1} for target in targets]
    return targets, rows


def replay_rows():
    targets = [index % 25 for index in range(96)]
    return targets, [{'split': 'train', 'native_font_verified': True, 'target': target,
        'family': short.FAMILIES[target], 'domain': 'android', 'view': 'native',
        'source_font_family': short.FAMILIES[target], 'font_face': f'face-{target}'}
        for target in targets]


def test_short_sampler_is_deterministic_balanced_and_cycles_lengths_and_sources():
    rows = short_rows(all_lengths=True)
    left = short.ShortSampler(deepcopy(rows), 41)
    right = short.ShortSampler(deepcopy(rows), 41)
    left_batches = [left.batch() for _ in range(36)]
    right_batches = [right.batch() for _ in range(36)]
    identity = lambda row: (row['family'], row['source_font_family'], row['glyph_count'],
                            row['domain'], row['font_face'], row['view'])
    assert [[identity(row) for row in batch] for batch in left_batches] == [
        [identity(row) for row in batch] for batch in right_batches]
    for batch in left_batches:
        assert len(batch) == 32
        assert sorted(row['family'] for row in batch if row['family'] != '__unknown__') == sorted(short.FAMILIES[:-1])
        assert sum(row['family'] == '__unknown__' for row in batch) == 8
    report = left.report()
    assert report['steps'] == 36 and report['rows'] == 36 * 32
    unknown_counts = {source: 0 for source in short.SHORT_UNKNOWN_SOURCES}
    for entry in report['counts']:
        if entry['family'] == '__unknown__':
            unknown_counts[entry['source_font_family']] += entry['rows']
    assert sum(unknown_counts.values()) == 36 * 8
    assert max(unknown_counts.values()) - min(unknown_counts.values()) == 0
    assert {entry['glyph_count'] for entry in report['counts']} == {1, 2, 3, 4}


@pytest.mark.parametrize('fault', ['missing_unknown', 'heldout_unknown', 'relabel', 'heldout_split',
                                   'unverified', 'glyph_count'])
def test_short_sampler_rejects_source_isolation_and_truth_corruption(fault):
    rows = short_rows()
    if fault == 'missing_unknown':
        rows = [row for row in rows if row.get('source_font_family') != short.SHORT_UNKNOWN_SOURCES[-1]]
    elif fault == 'heldout_unknown':
        row = next(row for row in rows if row['family'] == '__unknown__')
        row['source_font_family'] = 'Yusei Magic'
    elif fault == 'relabel':
        rows[0]['target'] = 1
    elif fault == 'heldout_split':
        rows[0]['split'] = 'calibration'
    elif fault == 'unverified':
        rows[0]['native_font_verified'] = False
    else:
        rows[0]['glyph_count'] = 5
    with pytest.raises(ValueError):
        short.ShortSampler(rows, 1)


def test_short_loss_matches_exact_ce_unknown_floor_and_beta_point_zero_five():
    torch = pytest.importorskip('torch')
    from torch.nn import functional as F
    from retention_confidence_floor_loss import unknown_floor_loss
    targets_list, rows = loss_rows()
    logits = torch.linspace(-1., 1., 32 * 25, dtype=torch.float64).reshape(32, 25).requires_grad_()
    ratios = torch.linspace(-.3, .4, 32, dtype=torch.float64).requires_grad_()
    targets = torch.tensor(targets_list, dtype=torch.int64)
    sizes = torch.linspace(.2, -.1, 32, dtype=torch.float64)
    weighted, font, size = short.short_loss(logits, ratios, targets, sizes, rows)
    known = targets != 24
    expected_font = (F.cross_entropy(logits[known], targets[known], label_smoothing=.03,
                                     reduction='sum')
                     + unknown_floor_loss(logits[~known], 24).sum()) / 32
    expected_size = F.smooth_l1_loss(ratios, sizes, beta=.05)
    assert torch.allclose(font, expected_font, atol=1e-12, rtol=1e-12)
    assert torch.allclose(size, expected_size, atol=1e-12, rtol=1e-12)
    assert torch.allclose(weighted, .5 * (expected_font + .2 * expected_size), atol=1e-12, rtol=1e-12)
    weighted.backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) > 0
    assert torch.isfinite(ratios.grad).all() and torch.count_nonzero(ratios.grad) == 32


@pytest.mark.parametrize('fault', ['target_shape', 'target_dtype', 'target_row', 'family_row',
                                   'nan_logits', 'nan_size', 'heldout_row'])
def test_short_loss_rejects_tensor_and_row_corruption(fault):
    torch = pytest.importorskip('torch')
    values, rows = loss_rows()
    logits = torch.zeros(32, 25); ratios = torch.zeros(32)
    targets = torch.tensor(values); sizes = torch.zeros(32)
    if fault == 'target_shape': targets = targets[:-1]
    elif fault == 'target_dtype': targets = targets.float()
    elif fault == 'target_row': rows[0]['target'] = 1
    elif fault == 'family_row': rows[0]['family'] = short.FAMILIES[1]
    elif fault == 'nan_logits': logits[0, 0] = float('nan')
    elif fault == 'nan_size': sizes[0] = float('nan')
    else: rows[0]['split'] = 'calibration'
    with pytest.raises((ValueError, RuntimeError)):
        short.short_loss(logits, ratios, targets, sizes, rows)


def metrics():
    return {'known_correct_coverage': .65, 'named_precision': .95, 'unknown_not_named_rate': .85,
            'per_domain': {'ios': {'known_correct_coverage': .75},
                           'android': {'known_correct_coverage': .55}}}


def test_exact_seven_calibration_regression_checks_pass_at_boundaries():
    baseline = metrics(); current = deepcopy(baseline)
    current['per_domain']['ios']['known_correct_coverage'] -= .005
    current['per_domain']['android']['known_correct_coverage'] -= .005
    baseline_short = {'known_views': 100, 'correct_named': 50, 'known_correct_coverage': .5,
                      'unknown_views': 50, 'unknown_falsely_named': 10}
    current_short = {**baseline_short, 'correct_named': 52, 'known_correct_coverage': .52}
    result = short.compare_r22(current, baseline, current_short, baseline_short)
    assert result['passed'] and len(result['checks']) == 7 and all(row['passed'] for row in result['checks'])


@pytest.mark.parametrize('failed_name', ['short_known_coverage', 'short_unknown_false_names',
    'all.known_correct_coverage', 'all.named_precision', 'all.unknown_not_named_rate',
    'ios.known_correct_coverage', 'android.known_correct_coverage'])
def test_each_additional_calibration_condition_is_independently_required(failed_name):
    baseline = metrics(); current = deepcopy(baseline)
    baseline_short = {'known_views': 100, 'correct_named': 50, 'known_correct_coverage': .5,
                      'unknown_views': 50, 'unknown_falsely_named': 10}
    current_short = {**baseline_short, 'correct_named': 52, 'known_correct_coverage': .52}
    if failed_name == 'short_known_coverage': current_short['known_correct_coverage'] = .519
    elif failed_name == 'short_unknown_false_names': current_short['unknown_falsely_named'] = 11
    elif failed_name.startswith('all.'):
        current[failed_name.split('.', 1)[1]] -= .001
    else:
        current['per_domain'][failed_name.split('.')[0]]['known_correct_coverage'] -= .006
    result = short.compare_r22(current, baseline, current_short, baseline_short)
    assert not result['passed']
    assert [row['name'] for row in result['checks'] if not row['passed']] == [failed_name]


def test_real_wide_network_combined_objective_has_finite_gradients_in_all_four_groups():
    torch = pytest.importorskip('torch')
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4); torch.manual_seed(7)
    model = WideRegionFontClassifier(25).train()
    replay_targets, old_rows = replay_rows()
    short_targets, small_rows = loss_rows()
    teacher = torch.full((96, 25), -3.)
    teacher[torch.arange(96), torch.tensor(replay_targets)] = 3.
    batch = {'images': torch.rand(128, 1, 64, 256), 'targets': torch.tensor(replay_targets),
             'sizes': torch.linspace(-.3, .3, 96), 'teacher': teacher,
             'rows': old_rows, 'short_targets': torch.tensor(short_targets),
             'short_sizes': torch.linspace(-.2, .2, 32), 'short_rows': small_rows}
    total, parts = trainer.objective(model, batch)
    assert torch.isfinite(total) and set(parts) == {'replay', 'replay_ce', 'teacher_kl',
                                                    'short_weighted', 'short_font', 'short_size'}
    total.backward()
    for group in ('trunk', 'style', 'family_head', 'size_head'):
        gradients = [parameter.grad for name, parameter in model.named_parameters()
                     if name.startswith(group + '.')]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert any(torch.count_nonzero(value) for value in gradients)


class StubSampler:
    def __init__(self, rows, seed=3):
        self.rows = rows
        self.rng = np.random.default_rng(seed)

    def batch(self):
        return list(self.rows)


def test_inputs_pair_each_replay_tile_with_same_index_r22_logits_and_no_short_teacher():
    pytest.importorskip('torch')
    sources = {}
    sampled = []
    for source_index, name in enumerate(('original', 'supplement', 'known')):
        tiles = np.full((32, 1, 64, 256), source_index / 3, dtype=np.float32)
        teacher = np.stack([np.full(25, source_index * 100 + index, np.float32) for index in range(32)])
        index = {}
        for tile_index in range(32):
            target = (source_index * 32 + tile_index) % 25
            raw = {'source_id': f'{name}-{tile_index}', 'region_id': f'r-{tile_index}',
                   'view': 'native', 'split': 'train', 'native_font_verified': True,
                   'tile_start': tile_index, 'tile_count': 1, 'target': target,
                   'family': short.FAMILIES[target], 'log_em_ratio': tile_index / 100}
            index[(raw['source_id'], raw['region_id'], raw['view'])] = raw
            marker = ({SUPPLEMENT_MARKER: True} if name == 'supplement' else
                      {KNOWN_SUPPLEMENT_MARKER: True} if name == 'known' else {})
            sampled.append({**raw, **marker})
        sources[name] = {'tiles': tiles, 'teacher_logits': teacher, 'index': index}
    targets, small = loss_rows()
    for index, row in enumerate(small):
        row.update(index=index, tile_start=index, tile_count=1, log_em_ratio=index / 50)
    short_data = {'rows': deepcopy(small), 'tiles': np.full((32, 1, 64, 256), .75, np.float32)}
    batch = trainer.inputs(sources, short_data, StubSampler(sampled), StubSampler(small), 'cpu')
    assert batch['images'].shape == (128, 1, 64, 256)
    assert batch['teacher'].shape == (96, 25)
    assert np.array_equal(batch['indices'], [(name, index) for name in ('original', 'supplement', 'known')
                                              for index in range(32)])
    assert batch['short_indices'] == list(range(32))
    assert len(batch['short_rows']) == 32 and len(batch['teacher']) == 96
    for source_index in range(3):
        expected = np.stack([np.full(25, source_index * 100 + index, np.float32) for index in range(32)])
        assert np.array_equal(batch['teacher'][source_index * 32:(source_index + 1) * 32].numpy(), expected)


def test_design_pins_fixed_final_schedule_runtime_and_both_dev_policies(tmp_path):
    args = SimpleNamespace(teacher=tmp_path / 'teacher', short=tmp_path / 'short')
    result = trainer.design(args, {'bound': 'sha'})
    assert result['steps'] == result['evaluation_step'] == 2400
    assert result['learning_rate'] == 5e-6 and result['minimum_learning_rate'] == 5e-7
    assert result['checkpoint_selection'] == 'fixed final step 2400; no intermediate CAL search'
    assert result['initializer_state_sha256'] == trainer.BASE_STATE_SHA
    assert result['objective']['replay_batch'] == 96 and result['objective']['short_batch'] == 32
    assert result['objective']['short_teacher'] is None and result['objective']['runtime_changes'] is False
    assert result['additional_acceptance']['original_46_checks_required'] is True
    assert result['additional_acceptance']['original_18_development_checks_required'] is True
    assert result['additional_acceptance']['r22_18_development_checks_also_required'] is True
    assert result['test_read'] is False and result['development_holdout_read'] is False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_files_bound_includes_short_native_proof_closure(monkeypatch, tmp_path):
    root = tmp_path / 'root'; base = tmp_path / 'r22'; teacher = tmp_path / 'teacher'; data = tmp_path / 'short'
    for folder in (root / 'training', base, teacher, data / 'train'):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ('train_unified_short_regions.py', 'short_region_objective.py',
                 'prepare_short_region_supplement.py'):
        (root / 'training' / name).write_text(name)
    for name in ('CALIBRATION_OUTPUTS.npz', 'CALIBRATION_DECISIONS.json', 'DEVELOPMENT_REGRESSION.json'):
        (base / name).write_text(name)
    (base / 'SELECTION.json').write_text(json.dumps({'bindings': {}}))
    retention = root / 'RETENTION_PLAN.json'; retention.write_text('{}')
    proof = root / 'native-proof.json'; proof.write_text('verified TRAIN proof')
    array = data / 'train/tiles.raw'; array.write_bytes(b'tiny')
    metadata = data / 'train/rows.json'; metadata.write_text('[]')
    (data / 'MANIFEST.json').write_text(json.dumps({'bindings': {str(proof.resolve()): sha(proof)}}))
    (data / 'train/MANIFEST.json').write_text(json.dumps({
        'array': {'path': 'tiles.raw'}, 'metadata': {'path': 'rows.json'}}))
    (teacher / 'CACHE_FREEZE.json').write_text('{}')
    teacher_array = teacher / 'original/logits.npy'; teacher_array.parent.mkdir(); teacher_array.write_bytes(b'logits')
    (teacher / 'CACHE_MANIFEST.json').write_text(json.dumps({'partitions': {
        'original': {'files': {'logits': {'path': 'original/logits.npy'}}}}}))
    monkeypatch.setattr(trainer, 'ROOT', root)
    monkeypatch.setattr(trainer, 'BASE_RUN', base)
    monkeypatch.setattr(trainer, 'RETENTION_PLAN', retention)
    monkeypatch.setattr(trainer, 'source_bindings', lambda datasets: {})
    result = trainer.files_bound(SimpleNamespace(teacher=teacher, short=data), {})
    assert result[str(proof.resolve())] == sha(proof)
    (base / 'SELECTION.json').write_text(json.dumps({'bindings': {str(proof.resolve()): '0' * 64}}))
    with pytest.raises(ValueError, match='conflict'):
        trainer.files_bound(SimpleNamespace(teacher=teacher, short=data), {})
    (base / 'SELECTION.json').write_text(json.dumps({'bindings': {}}))
    proof.write_text('tampered after short manifest')
    with pytest.raises(ValueError):
        trainer.files_bound(SimpleNamespace(teacher=teacher, short=data), {})


def test_calibration_success_is_recorded_but_never_becomes_release_promotion(
        monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    base = tmp_path / 'r22'; base.mkdir()
    np.savez(base / 'CALIBRATION_OUTPUTS.npz', logits=np.zeros((2, 25), np.float32),
             log_em_ratio=np.zeros(2, np.float32))
    retention = tmp_path / 'RETENTION_PLAN.json'; retention.write_text('{}')
    preflight = tmp_path / 'PREFLIGHT.json'
    preflight.write_text(json.dumps({'passed': True, 'bindings': {}, 'test_read': False}))
    args = SimpleNamespace(output=tmp_path / 'run', plan=tmp_path / 'PLAN.json',
                           teacher=tmp_path / 'teacher', short=tmp_path / 'short', device='cpu')
    monkeypatch.setattr(trainer, 'BASE_RUN', base)
    monkeypatch.setattr(trainer, 'RETENTION_PLAN', retention)
    monkeypatch.setattr(trainer, 'STEPS', 1)
    monkeypatch.setattr(trainer, 'files_bound', lambda args, datasets: {})

    model = torch.nn.Linear(1, 1, bias=False)
    base_metrics = metrics() | {'passed': False}
    base_selection = {'selected': {'metrics': deepcopy(base_metrics)}}
    report_stub = SimpleNamespace(report=lambda: {'synthetic': True})
    monkeypatch.setattr(trainer, 'context', lambda args:
        (model, {}, {}, {}, report_stub, report_stub, base_selection, {}))
    monkeypatch.setattr(trainer, 'parameter_groups', lambda state:
        {'all': float(next(iter(state.values())).sum())})
    replay_unknown_sources = [*short.SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong']
    replay = []
    for index in range(96):
        target = 24 if index < 16 else index % 24
        replay.append({'family': short.FAMILIES[target], 'target': target,
                       'source_font_family': replay_unknown_sources[index % 11]
                       if target == 24 else short.FAMILIES[target]})
    small_targets, small = loss_rows()
    batch = {'rows': replay, 'short_rows': small}
    monkeypatch.setattr(trainer, 'inputs', lambda *args: batch)
    monkeypatch.setattr(trainer, 'objective', lambda model, batch:
        ((model.weight ** 2).sum(), {name: (model.weight ** 2).sum() for name in
          ('replay', 'replay_ce', 'teacher_kl', 'short_weighted', 'short_font', 'short_size')}))
    cal = {'rows': [{}, {}], 'manifest_sha256': 'm', 'partition_sha256': 'p'}
    monkeypatch.setattr(trainer, 'load_split', lambda *args: cal)
    record = {'metrics': deepcopy(base_metrics), 'retention_checks': [{'passed': True}] * 46,
              'promotion_allowed': True}
    monkeypatch.setattr(trainer, 'evaluate_outputs', lambda *args:
        (deepcopy(record), [], [{}, {}]))
    short_results = iter((
        {'known_views': 100, 'correct_named': 50, 'known_correct_coverage': .5,
         'unknown_views': 50, 'unknown_falsely_named': 10},
        {'known_views': 100, 'correct_named': 52, 'known_correct_coverage': .52,
         'unknown_views': 50, 'unknown_falsely_named': 10}))
    monkeypatch.setattr(trainer, 'short_population', lambda *args: next(short_results))
    monkeypatch.setattr(trainer, 'infer', lambda *args:
        (np.zeros((2, 25), np.float32), np.zeros(2, np.float32)))
    plan = {**trainer.design(args, {}),
            'preflight': {'path': str(preflight), 'sha256': trainer.sha(preflight)}}
    args.plan.write_text(json.dumps(plan))
    trainer.train(args)
    selection = json.loads((args.output / 'SELECTION.json').read_text())
    report = json.loads((args.output / 'report.json').read_text())
    assert selection['calibration_promotion_allowed'] is True
    assert selection['promotion_allowed'] is False and selection['development_evaluated'] is False
    assert report['calibration_promotion_allowed'] is True and report['promotion_allowed'] is False
    assert report['status'] == 'CAL_PASSED_AWAITING_EXPORT_AND_DEVELOPMENT'
