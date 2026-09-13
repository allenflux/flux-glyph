"""Merged-weight b08 logits cache tests use only synthetic TRAIN tensors."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from training import cache_unified_weight_teacher as module


def fixture(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    root = tmp_path/'root'; root.mkdir(); monkeypatch.setattr(module, 'ROOT', root)
    for relative in module.REQUIRED_SOURCES:
        path = root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source '+relative)
    data_root = root/'merged-data'; (data_root/'train').mkdir(parents=True)
    tiles = np.random.default_rng(14).random((5, 1, 64, 256), dtype=np.float32)
    (data_root/'train/tiles.raw').write_bytes(tiles.astype('<f4').tobytes())
    families = list(module.FAMILIES); rows = []
    for start, count, family, face in ((0, 2, 'LXGW WenKai', 'LXGWWenKai-Regular'),
                                       (2, 1, 'LXGW WenKai', 'LXGWWenKai-Light'),
                                       (3, 2, 'WenQuanYi Micro Hei', 'WenQuanYiMicroHei')):
        rows.append({'split': 'train', 'original_split': 'train', 'family': family,
            'target': families.index(family), 'domain': 'android', 'source_font_family': family,
            'source_dataset': 'android_paired_known_supplement', 'native_font_verified': True,
            'font_face': face, 'font_file_sha256': f'{start+1:064x}', 'ttc_index': 0,
            'source_id': f'source-{start}', 'region_id': f'region-{start}', 'view': 'native',
            'tile_start': start, 'tile_count': count,
            'tiles_sha256': hashlib.sha256(tiles[start:start+count].tobytes()).hexdigest()})
    module.dump(data_root/'train/rows.json', rows)
    module.dump(data_root/'train/SOURCE_MAPPING.json', {'schema': 'synthetic-source-mapping'})
    proof = root/'proof.json'; proof.write_text('synthetic merged source proof')
    freeze = {'schema': 'flux-glyph-unified-known-weight-merged-freeze-v1', 'only_split': 'train'}
    module.dump(data_root/'PREPARATION_FREEZE.json', freeze)
    manifest = {**freeze, 'schema': module.DATA_SCHEMA, 'families': families,
        'bindings': {str(proof): module.sha(proof)}, 'sources': {}}
    module.dump(data_root/'MANIFEST.json', manifest)
    part = {'schema': module.DATA_SCHEMA, 'split': 'train', 'families': families,
        'root_manifest_sha256': module.sha(data_root/'MANIFEST.json'), 'views': len(rows), 'tiles': len(tiles),
        'shape': list(tiles.shape),
        'metadata': {'path': 'rows.json', 'sha256': module.sha(data_root/'train/rows.json')},
        'array': {'path': 'tiles.raw', 'sha256': module.sha(data_root/'train/tiles.raw')},
        'source_mapping': {'path': 'SOURCE_MAPPING.json', 'sha256': module.sha(data_root/'train/SOURCE_MAPPING.json')}}
    module.dump(data_root/'train/MANIFEST.json', part)
    mapped = np.memmap(data_root/'train/tiles.raw', mode='r', dtype='<f4', shape=tiles.shape)
    data = {'tiles': mapped, 'rows': rows, 'families': families, 'manifest': manifest, 'partition': part,
        'manifest_sha256': module.sha(data_root/'MANIFEST.json'),
        'partition_sha256': module.sha(data_root/'train/MANIFEST.json')}

    model = RegionFontClassifier(25).eval().requires_grad_(False)
    state = module.state_sha(model.state_dict()); groups = module.strict.parameter_groups(model.state_dict())
    teacher_root = root/'teacher'; teacher_root.mkdir()
    for name in ('model.pth', 'SELECTION.json', 'TRAINING_FREEZE.json'):
        (teacher_root/name).write_bytes(('teacher '+name).encode())
    teacher = {'checkpoint': module.entry(teacher_root/'model.pth'),
        'selection': module.entry(teacher_root/'SELECTION.json'),
        'training_freeze': module.entry(teacher_root/'TRAINING_FREEZE.json'),
        'state_sha256': state, 'parameter_groups_sha256': groups,
        'selected_step': 1500, 'optimizer_steps_executed': 3000}
    monkeypatch.setattr(module.strict, 'TEACHER_CHECKPOINT_SHA', teacher['checkpoint']['sha256'])
    monkeypatch.setattr(module.strict, 'TEACHER_SELECTION_SHA', teacher['selection']['sha256'])
    monkeypatch.setattr(module.strict, 'TEACHER_STATE_SHA', state)
    identity = module.cache_identity(data_root, data, teacher, 'cpu')
    return data_root, data, teacher, model, identity, root/'teacher-cache'


def test_cpu_extraction_is_exact_logits_only_and_preserves_teacher(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(2)
    data_root, data, teacher, model, identity, output = fixture(monkeypatch, tmp_path)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode(): expected, _ = model(torch.from_numpy(np.array(data['tiles'], copy=True)))
    arrays, manifest, cache = module.extract_cache(output, model, data, identity, 'cpu')
    assert set(arrays) == {'base_logits'} and np.array_equal(arrays['base_logits'], expected.numpy())
    assert isinstance(arrays['base_logits'], np.memmap) and not arrays['base_logits'].flags.writeable
    assert set(manifest) >= {'identity', 'base_logits', 'teacher_state_before_sha256', 'teacher_state_after_sha256'}
    assert manifest['teacher_state_before_sha256'] == manifest['teacher_state_after_sha256'] == teacher['state_sha256']
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert cache == module.entry(output/'CACHE_MANIFEST.json')
    assert not any((output/name).exists() for name in ('features.npy', 'log_em_ratio.npy', 'calibration', 'development', 'test'))
    bindings = {}; loaded = module.load_cache(output, data, teacher, bindings)
    assert np.array_equal(loaded[0]['base_logits'], expected.numpy()) and loaded[1] == manifest and loaded[2] == cache
    assert bindings[cache['path']] == cache['sha256']
    assert all(bindings[path] == digest for path, digest in identity['bindings'].items())
    assert identity['dataset']['manifest']['path'] == str((data_root/'MANIFEST.json').resolve())


@pytest.mark.parametrize('fault', ['row_order', 'tile_bytes', 'cache_bytes', 'cache_shape',
    'teacher', 'freeze_optimizer', 'source', 'binding_conflict'])
def test_loaded_data_cache_teacher_and_source_tampering_fails(monkeypatch, tmp_path, fault):
    _, data, teacher, model, identity, output = fixture(monkeypatch, tmp_path)
    module.extract_cache(output, model, data, identity, 'cpu')
    bindings = {}
    if fault == 'row_order': data['rows'][1]['tile_start'] = 0
    elif fault == 'tile_bytes':
        data['tiles'] = np.array(data['tiles'], copy=True)
        data['tiles'][0, 0, 0, 0] = .5
    elif fault == 'cache_bytes': (output/'base_logits.npy').write_bytes(b'changed')
    elif fault == 'cache_shape':
        manifest = module.read(output/'CACHE_MANIFEST.json'); manifest['base_logits']['shape'][0] -= 1
        module.dump(output/'CACHE_MANIFEST.json', manifest)
    elif fault == 'teacher': teacher = {**teacher, 'state_sha256': '0'*64}
    elif fault == 'freeze_optimizer':
        freeze = module.read(output/'CACHE_FREEZE.json'); freeze['optimizer_steps_executed'] = 1
        module.dump(output/'CACHE_FREEZE.json', freeze)
        manifest = module.read(output/'CACHE_MANIFEST.json')
        manifest['cache_freeze_sha256'] = module.sha(output/'CACHE_FREEZE.json'); module.dump(output/'CACHE_MANIFEST.json', manifest)
    elif fault == 'source': (module.ROOT/module.REQUIRED_SOURCES[0]).write_text('changed')
    else: bindings[next(iter(identity['bindings']))] = 'f'*64
    with pytest.raises((ValueError, OSError, EOFError)):
        module.load_cache(output, data, teacher, bindings)


def test_fresh_output_is_mandatory(monkeypatch, tmp_path):
    _, data, _, model, identity, output = fixture(monkeypatch, tmp_path)
    output.mkdir()
    with pytest.raises(ValueError, match='fresh path'):
        module.extract_cache(output, model, data, identity, 'cpu')


@pytest.mark.parametrize('fault', ['heldout', 'relabel', 'offset', 'hash', 'nan', 'range'])
def test_only_exact_merged_train_order_and_pixels_are_accepted(monkeypatch, tmp_path, fault):
    _, data, *_ = fixture(monkeypatch, tmp_path)
    if fault == 'heldout': data['rows'][0]['split'] = 'calibration'
    elif fault == 'relabel': data['rows'][0]['target'] = 24
    elif fault == 'offset': data['rows'][1]['tile_start'] = 1
    elif fault == 'hash': data['rows'][0]['tiles_sha256'] = '0'*64
    elif fault == 'nan':
        data['tiles'] = np.array(data['tiles'], copy=True)
        data['tiles'][0, 0, 0, 0] = np.nan
    else:
        data['tiles'] = np.array(data['tiles'], copy=True)
        data['tiles'][0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError): module.validate_data(data)


def test_cli_defaults_are_fresh_merged_weight_paths():
    args = module.parser().parse_args([])
    assert args.data.name == 'merged-data' and args.output.name == 'teacher-cache'
    assert args.data.parent == args.output.parent
    assert args.output.parent.name == 'paired-wenkai-weight-capture-v1'
    assert args.checkpoint.parent.name == 'run-source-balanced-v1' and args.device == 'mps'
