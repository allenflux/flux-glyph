import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from training import cache_unified_student_teacher as cache

torch = pytest.importorskip('torch')
from region_network import RegionFontClassifier


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(2); torch.manual_seed(42)
    model = RegionFontClassifier(25).eval().requires_grad_(False)
    # A nonzero size head is part of full state, although only logits are cached.
    with torch.no_grad(): model.size_head.bias.fill_(.3)
    state = cache.state_sha(model.state_dict()); groups = cache.parameter_groups(model.state_dict())
    folder = tmp_path/'teacher'; folder.mkdir()
    freeze = {'schema': 'flux-glyph-unified-retention-source-balanced-protocol-v1',
        'architecture': cache.ARCHITECTURE, 'families': list(cache.FAMILIES), 'optimizer_steps_executed': 3000,
        'initial_parameter_groups_sha256': {k: '0'*64 for k in groups},
        'test_read': False, 'development_holdout_read': False, 'bindings': {}}
    cache.dump(folder/'TRAINING_FREEZE.json', freeze)
    history = [{'step': step, 'promotion_allowed': False, 'retention_deficit': abs(1500-step),
        'metrics': {'correct_named': 10, 'wrong_named': 0}, 'calibration_nll': .1} for step in range(500, 3001, 500)]
    selection = {**freeze, 'schema': 'flux-glyph-unified-retention-source-balanced-selection-v1',
        'history': history, 'selected': history[2], 'state_after_sha256': state,
        'selected_parameter_groups_sha256': groups, 'final_parameter_groups_sha256': groups,
        'training_protocol_sha256': cache.sha(folder/'TRAINING_FREEZE.json')}
    cache.dump(folder/'SELECTION.json', selection)
    # Historical final checkpoints have no redundant top-level step field.
    checkpoint = {'state_dict': model.state_dict(), 'families': list(cache.FAMILIES),
        'architecture': cache.ARCHITECTURE, 'selection_sha256': cache.sha(folder/'SELECTION.json')}
    torch.save(checkpoint, folder/'model.pth')
    for name, value in [('TEACHER_CHECKPOINT_SHA', cache.sha(folder/'model.pth')),
        ('TEACHER_SELECTION_SHA', cache.sha(folder/'SELECTION.json')),
        ('TEACHER_FREEZE_SHA', cache.sha(folder/'TRAINING_FREEZE.json')), ('TEACHER_STATE_SHA', state)]:
        monkeypatch.setattr(cache, name, value)
    roots = {}; pins = {}; blocked = tmp_path/'calibration/tiles.raw'
    for index, source in enumerate(cache.PARTITIONS):
        root = tmp_path/source; (root/'train').mkdir(parents=True); roots[source] = root
        # A historical closure entry must stay metadata-only; this path does not exist.
        manifest = {'schema': cache.DATA_SCHEMAS[source], 'families': list(cache.FAMILIES),
            'bindings': {str(blocked): 'b'*64}}
        if source != 'original':
            manifest['only_split'] = 'train'
            cache.dump(root/'PREPARATION_FREEZE.json', manifest)
            manifest['preparation_freeze_sha256'] = cache.sha(root/'PREPARATION_FREEZE.json')
        cache.dump(root/'MANIFEST.json', manifest)
        rng = np.random.default_rng(index)
        tiles = rng.random((3, 1, 64, 256), dtype=np.float32)
        tiles.tofile(root/'train/tiles.raw'); rows = []
        for rowindex, (start, count) in enumerate(((0, 1), (1, 2))):
            target = 24 if source == 'supplement' else (10+rowindex if source == 'known' else rowindex)
            family = cache.FAMILIES[target]
            row = {'family': family, 'target': target, 'source_font_family': family,
                'source_id': source+str(rowindex), 'region_id': 'r'+str(rowindex), 'view': 'native',
                'native_font_verified': True, 'split': 'train', 'original_split': 'train', 'domain': 'android',
                'source_dataset': 'fixture', 'tile_start': start, 'tile_count': count,
                'tiles_sha256': digest(tiles[start:start+count])}
            if source == 'supplement':
                row.update(source_dataset='android_new_unknown_supplement', source_font_family='Zhuque Fangsong')
            if source == 'known': row['source_dataset'] = 'android_paired_known_supplement'
            rows.append(row)
        cache.dump(root/'train/rows.json', rows)
        part = {'schema': manifest['schema'], 'split': 'train', 'families': list(cache.FAMILIES),
            'root_manifest_sha256': cache.sha(root/'MANIFEST.json'), 'views': 2, 'tiles': 3,
            'shape': [3, 1, 64, 256], 'metadata': {'path': 'rows.json', 'sha256': cache.sha(root/'train/rows.json')},
            'array': {'path': 'tiles.raw', 'sha256': cache.sha(root/'train/tiles.raw')}}
        cache.dump(root/'train/MANIFEST.json', part)
        pins[source] = (part['root_manifest_sha256'], cache.sha(root/'train/MANIFEST.json'), 2, 3)
    monkeypatch.setattr(cache, 'DATA_PINS', pins)
    identity = cache.cache_identity(folder/'model.pth', roots, 'cpu')
    datasets = {name: cache.load_train_partition(name, identity['partitions'][name]) for name in cache.PARTITIONS}
    return model, identity, datasets, tmp_path/'cache', blocked


def test_three_source_actual_logits_and_freeze_before_forward(fixture):
    model, identity, datasets, output, blocked = fixture
    before = cache.state_sha(model.state_dict()); seen = []
    expected = {}
    with torch.inference_mode():
        for name, data in datasets.items():
            expected[name] = model(torch.from_numpy(np.array(data['tiles'], copy=True)))[0].numpy()
    def verify_before(module, inputs):
        frozen = cache.read(output/'CACHE_FREEZE.json')
        assert frozen['identity'] == identity
        seen.append(len(inputs[0]))
    hook = model.register_forward_pre_hook(verify_before)
    arrays, manifest = cache.extract_cache(output, model, datasets, identity, 'cpu'); hook.remove()
    assert seen == [3, 3, 3] and manifest['tiles_inferred'] == 9
    assert manifest['teacher_state_before_sha256'] == manifest['teacher_state_after_sha256'] == before
    assert set(arrays) == set(cache.PARTITIONS)
    for name in cache.PARTITIONS:
        assert set(arrays[name]) == {'base_logits'}
        assert isinstance(arrays[name]['base_logits'], np.memmap)
        assert not arrays[name]['base_logits'].flags.writeable
        np.testing.assert_array_equal(arrays[name]['base_logits'], expected[name])
    assert not blocked.exists() and str(blocked) not in identity['bindings']
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    cache.load_cache(output, identity)


def test_exact_checkpoint_without_redundant_step_loads(fixture):
    model, identity, *_ = fixture
    loaded = cache.load_teacher(identity)
    assert cache.state_sha(loaded.state_dict()) == cache.state_sha(model.state_dict())


@pytest.mark.parametrize('field,value', [
    ('historical_logits_reused', True), ('teacher_logits_recomputed', False), ('optimizer_steps_executed', 1),
    ('calibration_images_read', True), ('outputs', ['features', 'base_logits']), ('total_tiles', 10)])
def test_reject_contract_mutation(fixture, field, value):
    _, identity, *_ = fixture
    identity[field] = value
    with pytest.raises(ValueError): cache.validate_identity(identity)


@pytest.mark.parametrize('field,value', [('selected_step', 1000), ('state_sha256', 'a'*64),
    ('optimizer_steps_executed', 1500)])
def test_reject_wrong_teacher(fixture, field, value):
    _, identity, *_ = fixture
    identity['teacher'][field] = value
    with pytest.raises(ValueError): cache.validate_identity(identity)


@pytest.mark.parametrize('field,value', [('tile_start', 1), ('tile_count', 0), ('split', 'calibration'),
    ('native_font_verified', False), ('target', True), ('view', 'unknown')])
def test_reject_wrong_row_identity(fixture, field, value):
    _, identity, datasets, *_ = fixture
    rows = copy.deepcopy(datasets['original']['rows']); rows[0][field] = value
    with pytest.raises(ValueError): cache.validate_rows(rows, list(cache.FAMILIES), 3, 'original')


def test_wrong_pixels_fail_before_cache_created(fixture):
    model, identity, datasets, output, _ = fixture
    datasets['known']['tiles'] = np.array(datasets['known']['tiles'], copy=True)
    datasets['known']['tiles'][1, 0, 0, 0] += .01
    with pytest.raises(ValueError): cache.extract_cache(output, model, datasets, identity, 'cpu')
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['training', 'grad', 'state'])
def test_not_frozen_complete_teacher_fails_before_cache(fixture, mutation):
    model, identity, datasets, output, _ = fixture
    if mutation == 'training': model.train()
    elif mutation == 'grad': next(model.parameters()).requires_grad_(True)
    else:
        with torch.no_grad(): model.size_head.bias.add_(1)
    with pytest.raises(ValueError): cache.extract_cache(output, model, datasets, identity, 'cpu')
    assert not output.exists()


def test_changed_output_bytes_rejected(fixture):
    model, identity, datasets, output, _ = fixture
    cache.extract_cache(output, model, datasets, identity, 'cpu')
    path = output/'known/base_logits.npy'; arr = np.load(path); arr[0, 0] += 1; np.save(path, arr)
    with pytest.raises(ValueError): cache.load_cache(output)


def test_partial_cache_never_reused_or_overwritten(fixture):
    model, identity, datasets, output, _ = fixture
    output.mkdir(); (output/'CACHE_FREEZE.json').write_text('{}')
    with pytest.raises(FileNotFoundError): cache.extract_cache(output, model, datasets, identity, 'cpu')
    assert (output/'CACHE_FREEZE.json').read_text() == '{}'
