"""Resume only from the same frozen source run and CAL baseline."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import continue_sans_verifier as module


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def source(tmp_path, monkeypatch):
    args = SimpleNamespace(source_run=tmp_path/'source', data=tmp_path/'views', primary=tmp_path/'primary',
                           checkpoint=tmp_path/'teacher.pth', output=tmp_path/'new-run')
    root = tmp_path/'repo'
    monkeypatch.setattr(module, 'ROOT', root)
    families = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
                'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']
    for path in (args.checkpoint, args.primary/'model.onnx', args.primary/'rejection.onnx', args.source_run/'verifier.pth',
                 *(root/'training'/name for name in ('train_sans_verifier.py', 'prepare_sans_views.py',
                    'region_network.py', 'network.py', 'train_regions.py'))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('stand-in source: '+str(path.name)).encode())
    dump(args.data/'MANIFEST.json', {'families': families})
    dump(args.data/'train/MANIFEST.json', {'split': 'train'})
    dump(args.data/'calibration/MANIFEST.json', {'split': 'calibration', 'rows': 1, 'test_pixels_opened': False})
    dump(args.primary/'metadata.json', {'algorithm': 'region-cnn64x256-rejection-v2', 'families': families[:-1],
        'model': {'path': 'model.onnx', 'sha256': module.sha(args.primary/'model.onnx')},
        'rejection': {'model': {'path': 'rejection.onnx', 'sha256': module.sha(args.primary/'rejection.onnx')}}})
    dump(args.source_run/'BASELINE.json', {'families': families[:-1], 'records': [{'passed': True, 'known': True}]})
    dump(args.source_run/'CALIBRATION_DECISIONS.json', {'families': families, 'policy': module.POLICY, 'records': [{'passed': False}]})
    required = [args.data/'MANIFEST.json', args.data/'train/MANIFEST.json', args.data/'calibration/MANIFEST.json',
                args.checkpoint, args.primary/'metadata.json', args.primary/'model.onnx', args.primary/'rejection.onnx',
                *(root/'training'/name for name in ('train_sans_verifier.py', 'prepare_sans_views.py',
                                                   'region_network.py', 'network.py'))]
    selected = {'step': 4000, 'metrics': {'passed': False}}
    selection = {'schema': 'flux-glyph-sans-verifier-training-v1', 'policy': copy.deepcopy(module.POLICY),
        'families': families, 'test_read': False, 'optimizer_steps_executed': 4000, 'selected': selected,
        'history': [copy.deepcopy(selected)], 'bindings': {str(path): module.sha(path) for path in required},
        'baseline_sha256': module.sha(args.source_run/'BASELINE.json'),
        'calibration_decisions_sha256': module.sha(args.source_run/'CALIBRATION_DECISIONS.json')}
    checkpoint = {'families': families}
    def rebind():
        dump(args.source_run/'SELECTION.json', selection)
        checkpoint['selection_sha256'] = module.sha(args.source_run/'SELECTION.json')
        dump(args.source_run/'POLICY.json', {'policy': module.POLICY, 'test_read': False,
            'bindings': selection['bindings'], 'steps': selection['optimizer_steps_executed']})
    rebind()
    return SimpleNamespace(args=args, root=root, selection=selection, checkpoint=checkpoint, rebind=rebind)


def validate(case):
    return module.validate_resume(case.args, case.checkpoint)


def test_frozen_baseline_reused_without_opening_arrays_or_running_any_model(source, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Resume validation must not read any arrays'))
    before = {path: path.read_bytes() for path in source.args.source_run.rglob('*') if path.is_file()}
    selection, baseline, decisions, bindings = validate(source)
    assert selection == source.selection and baseline['records'] == [{'passed': True, 'known': True}]
    assert decisions['policy'] == module.POLICY and not source.args.output.exists()
    for name in ('SELECTION.json', 'verifier.pth', 'CALIBRATION_DECISIONS.json', 'BASELINE.json', 'POLICY.json'):
        path = source.args.source_run/name
        assert bindings[str(path)] == module.sha(path)
    assert bindings[str(Path(module.__file__).resolve())] == module.sha(module.__file__)
    assert bindings[str(source.root/'training/train_sans_verifier.py')] == module.sha(source.root/'training/train_sans_verifier.py')
    assert before == {path: path.read_bytes() for path in source.args.source_run.rglob('*') if path.is_file()}


@pytest.mark.parametrize('asset', ['baseline', 'decisions', 'selection', 'source', 'teacher', 'primary', 'rejection', 'views', 'calibration'])
def test_changed_source_or_cal_cache_is_rejected(source, asset):
    path = {'baseline': source.args.source_run/'BASELINE.json', 'decisions': source.args.source_run/'CALIBRATION_DECISIONS.json',
            'selection': source.args.source_run/'SELECTION.json', 'source': source.root/'training/train_sans_verifier.py',
            'teacher': source.args.checkpoint, 'primary': source.args.primary/'model.onnx',
            'rejection': source.args.primary/'rejection.onnx', 'views': source.args.data/'MANIFEST.json',
            'calibration': source.args.data/'calibration/MANIFEST.json'}[asset]
    path.write_bytes(path.read_bytes() + b'\nchanged')
    with pytest.raises((ValueError, json.JSONDecodeError)): validate(source)


@pytest.mark.parametrize('fault', ['test_read', 'policy', 'checkpoint_families', 'missing_binding', 'history', 'selected_step'])
def test_rehashed_invalid_lineage_cannot_reuse_baseline(source, fault):
    if fault == 'test_read': source.selection['test_read'] = True
    elif fault == 'policy': source.selection['policy']['gates']['min_score'] = .4
    elif fault == 'checkpoint_families': source.checkpoint['families'] = list(reversed(source.checkpoint['families']))
    elif fault == 'missing_binding': source.selection['bindings'].pop(str(source.args.checkpoint))
    elif fault == 'history': source.selection['history'] = []
    elif fault == 'selected_step': source.selection['selected']['step'] = 4001
    source.rebind()
    with pytest.raises(ValueError): validate(source)


def test_identical_bytes_in_an_unbound_cal_directory_are_not_the_same_cache(source, tmp_path):
    other = tmp_path/'other'
    for path in source.args.data.rglob('*'):
        if path.is_file():
            target = other/path.relative_to(source.args.data)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    source.args.data = other
    with pytest.raises(ValueError, match='differ'): validate(source)


def sampling_rows():
    return [{'split': 'train', 'target': target, 'domain': domain, 'view': view, 'tile_start': index*2,
             'tile_count': 2, 'source_id': 'source-'+str(index)}
            for index, (target, domain, view) in enumerate((target, domain, view)
                for target in range(9) for domain in ('anchor', 'new_native') for view in module.VIEWS)]


def test_sampler_remains_balanced_by_family_domain_and_requested_view_distribution():
    rows = sampling_rows()
    pools = module.training_pools(rows, 9)
    rng = np.random.default_rng(module.SEED)
    family_counts, view_counts, domain_counts = {}, {}, {}
    for step in range(1, 101):
        batch = module.sample_rows(pools, 9, step, rng)
        assert len(batch) == 64
        for row, tile in batch:
            assert row['tile_start'] <= tile < row['tile_start']+row['tile_count'] and row['split'] == 'train'
            for counts, key in ((family_counts, row['target']), (view_counts, row['view']), (domain_counts, row['domain'])):
                counts[key] = counts.get(key, 0)+1
    assert max(family_counts.values())-min(family_counts.values()) <= 1
    assert abs(view_counts['native']/6400-.4) < .025
    assert all(abs(view_counts[view]/6400-.2) < .025 for view in module.VIEWS if view != 'native')
    assert abs(domain_counts['anchor']/6400-.5) < .025
    assert module.sample_rows(pools, 9, 1, np.random.default_rng(42)) == module.sample_rows(pools, 9, 1, np.random.default_rng(42))


@pytest.mark.parametrize('fault', ['test', 'calibration', 'missing_view', 'bad_target'])
def test_sampler_never_accepts_calibration_test_or_incomplete_training_pools(fault):
    rows = sampling_rows()
    if fault in ('test', 'calibration'): rows[0]['split'] = fault
    elif fault == 'missing_view': rows = [row for row in rows if row['view'] != 'half']
    else: rows[0]['target'] = 9
    with pytest.raises(ValueError): module.training_pools(rows, 9)
