"""R22 TRAIN-teacher cache boundaries using only tiny synthetic TRAIN metadata/tensors."""
from copy import deepcopy
import json

import numpy as np
import pytest

from training import cache_r22_train_teacher as module


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def make_dataset(root, name, count=3):
    train = root / 'train'
    train.mkdir(parents=True)
    tiles = np.arange(count * 64 * 256, dtype=np.float32).reshape(count, 1, 64, 256)
    tiles /= tiles.max()
    (train / 'tiles.raw').write_bytes(tiles.astype('<f4').tobytes())
    rows = [
        {'split': 'train', 'native_font_verified': True, 'tile_start': 0, 'tile_count': 1,
         'target': 0, 'family': module.FAMILIES[0], 'source_id': f'{name}-0',
         'region_id': f'{name}-r0', 'view': 'native'},
        {'split': 'train', 'native_font_verified': True, 'tile_start': 1, 'tile_count': count - 1,
         'target': 1, 'family': module.FAMILIES[1], 'source_id': f'{name}-1',
         'region_id': f'{name}-r1', 'view': 'half'},
    ]
    dump(train / 'rows.json', rows)
    part = {'schema': 'synthetic-partition', 'split': 'train', 'tiles': count,
            'metadata': {'path': 'rows.json', 'sha256': module.sha(train / 'rows.json')},
            'array': {'path': 'tiles.raw', 'sha256': module.sha(train / 'tiles.raw')}}
    dump(train / 'MANIFEST.json', part)
    dump(root / 'MANIFEST.json', {'schema': 'synthetic-root', 'split': 'train', 'name': name})
    mapped = np.memmap(train / 'tiles.raw', mode='r', dtype='<f4', shape=tiles.shape)
    return {'families': list(module.FAMILIES), 'rows': rows, 'tiles': mapped,
            'partition': part, 'manifest': json.loads((root / 'MANIFEST.json').read_text())}


def fixture(monkeypatch, tmp_path):
    datasets = {}
    sources = {}
    for index, name in enumerate(('original', 'supplement', 'known')):
        root = tmp_path / name
        data = make_dataset(root, name, 3 + index)
        datasets[name] = data
        sources[name] = (root, module.sha(root / 'MANIFEST.json'), len(data['tiles']))
    monkeypatch.setattr(module, 'SOURCES', sources)
    if hasattr(module, 'TOTAL_TILES'):
        monkeypatch.setattr(module, 'TOTAL_TILES', sum(value[2] for value in sources.values()))
    return datasets, sources


def make_cache(tmp_path, datasets):
    output = tmp_path / 'cache'
    output.mkdir()
    bindings = module.source_bindings(datasets)
    freeze = {'schema': 'flux-glyph-r22-train-teacher-freeze-v1', 'bindings': bindings,
              'checkpoint_sha256': module.BASE_SHA, 'selection_sha256': module.BASE_SELECTION_SHA,
              'state_sha256': module.BASE_STATE_SHA, 'architecture': module.ARCHITECTURE,
              'families': list(module.FAMILIES), 'device': 'cpu',
              'libraries': {'torch': 'synthetic', 'numpy': np.__version__, 'python': 'synthetic'},
              'batch_size': 128, 'optimizer_steps': 0, 'calibration_read': False,
              'development_read': False, 'test_read': False, 'model_count': 1,
              'scope': 'Only original, unknown-supplement and merged-known TRAIN image tiles.'}
    dump(output / 'CACHE_FREEZE.json', freeze)
    partitions = {}
    for index, (name, data) in enumerate(datasets.items()):
        folder = output / name
        folder.mkdir()
        logits = np.arange(len(data['tiles']) * 25, dtype=np.float32).reshape(-1, 25) + index
        ratios = np.linspace(-.5, .5, len(data['tiles']), dtype=np.float32)
        np.save(folder / 'logits.npy', logits)
        np.save(folder / 'log_em_ratio.npy', ratios)
        partitions[name] = {'tiles': len(data['tiles']),
            'partition_sha256': module.sha(module.SOURCES[name][0] / 'train/MANIFEST.json'),
            'files': {key: {'path': f'{name}/{key}.npy',
                            'sha256': module.sha(folder / f'{key}.npy')}
                      for key in ('logits', 'log_em_ratio')}}
    manifest = {**freeze, 'schema': 'flux-glyph-r22-train-teacher-cache-v1',
                'freeze_sha256': module.sha(output / 'CACHE_FREEZE.json'),
                'partitions': partitions, 'total_tiles': sum(len(x['tiles']) for x in datasets.values()),
                'weights_unchanged': True, 'seconds': 0.0}
    dump(output / 'CACHE_MANIFEST.json', manifest)
    return output


def test_exact_readonly_train_mappings_are_accepted(monkeypatch, tmp_path):
    datasets, _ = fixture(monkeypatch, tmp_path)
    module.validate_datasets(datasets)
    assert all(isinstance(data['tiles'], np.memmap) and not data['tiles'].flags.writeable
               for data in datasets.values())


@pytest.mark.parametrize('fault', ['missing_source', 'row_order', 'row_relabel', 'partition',
                                   'writable', 'wrong_path', 'root_manifest'])
def test_in_memory_order_partition_path_and_source_hash_corruption_rejected(
        monkeypatch, tmp_path, fault):
    datasets, sources = fixture(monkeypatch, tmp_path)
    if fault == 'missing_source':
        datasets.pop('known')
    elif fault == 'row_order':
        datasets['original']['rows'] = list(reversed(datasets['original']['rows']))
    elif fault == 'row_relabel':
        datasets['original']['rows'][0] = {**datasets['original']['rows'][0], 'target': 2}
    elif fault == 'partition':
        datasets['supplement']['partition'] = {**datasets['supplement']['partition'], 'split': 'calibration'}
    elif fault == 'writable':
        root = sources['known'][0]
        datasets['known']['tiles'] = np.memmap(root / 'train/tiles.raw', mode='r+', dtype='<f4',
                                               shape=datasets['known']['tiles'].shape)
    elif fault == 'wrong_path':
        root = sources['known'][0]
        other = root / 'train/other.raw'
        other.write_bytes((root / 'train/tiles.raw').read_bytes())
        datasets['known']['tiles'] = np.memmap(other, mode='r', dtype='<f4',
                                               shape=datasets['known']['tiles'].shape)
    else:
        (sources['original'][0] / 'MANIFEST.json').write_text('changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        module.validate_datasets(datasets)


@pytest.mark.parametrize('fault', ['offset', 'relabel', 'heldout', 'unverified', 'orphan'])
def test_load_training_rejects_nontrain_relabel_and_offset_corruption(monkeypatch, tmp_path, fault):
    datasets, _ = fixture(monkeypatch, tmp_path)
    target = datasets['supplement']
    if fault == 'offset':
        target['rows'][1]['tile_start'] = 2
    elif fault == 'relabel':
        target['rows'][0]['family'] = module.FAMILIES[2]
    elif fault == 'heldout':
        target['rows'][0]['split'] = 'calibration'
    elif fault == 'unverified':
        target['rows'][0]['native_font_verified'] = False
    else:
        target['rows'][-1]['tile_count'] -= 1
    # Keep disk metadata equal so this exercises load_training's semantic checks,
    # rather than only the in-memory-versus-disk identity check.
    dump(module.SOURCES['supplement'][0] / 'train/rows.json', target['rows'])
    target['partition']['metadata']['sha256'] = module.sha(
        module.SOURCES['supplement'][0] / 'train/rows.json')
    dump(module.SOURCES['supplement'][0] / 'train/MANIFEST.json', target['partition'])
    monkeypatch.setattr(module, 'load_split', lambda root, split: datasets['original'])
    monkeypatch.setattr(module, 'load_supplement', lambda root: datasets['supplement'])
    monkeypatch.setattr(module, 'load_merged', lambda root: datasets['known'])
    with pytest.raises(ValueError):
        module.load_training()


def test_write_new_refuses_to_replace_an_existing_cache_file(tmp_path):
    path = tmp_path / 'CACHE_FREEZE.json'
    path.write_text('preserve')
    with pytest.raises(FileExistsError):
        module.write_new(path, {'changed': True})
    assert path.read_text() == 'preserve'


def test_load_cache_positive_roundtrip_is_exact_and_readonly(monkeypatch, tmp_path):
    datasets, _ = fixture(monkeypatch, tmp_path)
    output = make_cache(tmp_path, datasets)
    arrays, manifest = module.load_cache(output, datasets)
    assert set(arrays) == set(datasets) == set(manifest['partitions'])
    assert manifest['total_tiles'] == module.TOTAL_TILES == 12
    for name, values in arrays.items():
        assert set(values) == {'logits', 'log_em_ratio'}
        assert values['logits'].shape == (len(datasets[name]['tiles']), 25)
        assert values['log_em_ratio'].shape == (len(datasets[name]['tiles']),)
        assert all(isinstance(value, np.memmap) and not value.flags.writeable
                   for value in values.values())


@pytest.mark.parametrize('fault', ['row_order', 'cache_bytes', 'cache_shape', 'selection', 'state',
                                   'architecture', 'freeze_schema', 'source_bytes', 'extra_partition',
                                   'total_tiles', 'optimizer', 'heldout_claim'])
def test_load_cache_rejects_mapping_identity_hash_and_manifest_corruption(
        monkeypatch, tmp_path, fault):
    datasets, sources = fixture(monkeypatch, tmp_path)
    output = make_cache(tmp_path, datasets)
    manifest_path = output / 'CACHE_MANIFEST.json'
    freeze_path = output / 'CACHE_FREEZE.json'
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if fault == 'row_order':
        datasets['known']['rows'] = list(reversed(datasets['known']['rows']))
    elif fault == 'cache_bytes':
        (output / 'original/logits.npy').write_bytes(b'changed')
    elif fault == 'cache_shape':
        np.save(output / 'known/logits.npy', np.zeros((len(datasets['known']['tiles']), 24), np.float32))
        manifest['partitions']['known']['files']['logits']['sha256'] = module.sha(output / 'known/logits.npy')
    elif fault in ('selection', 'state', 'architecture'):
        key = {'selection': 'selection_sha256', 'state': 'state_sha256',
               'architecture': 'architecture'}[fault]
        freeze[key] = manifest[key] = 'wrong'
    elif fault == 'freeze_schema':
        freeze['schema'] = 'wrong-schema'
    elif fault == 'source_bytes':
        raw = sources['original'][0] / 'train/tiles.raw'
        with raw.open('r+b') as stream:
            stream.write(b'changed')
    elif fault == 'extra_partition':
        manifest['partitions']['calibration'] = deepcopy(manifest['partitions']['original'])
    elif fault == 'total_tiles':
        manifest['total_tiles'] += 1
    elif fault == 'optimizer':
        freeze['optimizer_steps'] = manifest['optimizer_steps'] = 1
    else:
        freeze['test_read'] = manifest['test_read'] = True
    if freeze != json.loads(freeze_path.read_text()):
        dump(freeze_path, freeze)
        manifest['freeze_sha256'] = module.sha(freeze_path)
    dump(manifest_path, manifest)
    with pytest.raises((ValueError, OSError, EOFError)):
        module.load_cache(output, datasets)
