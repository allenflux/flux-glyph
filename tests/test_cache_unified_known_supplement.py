"""TRAIN-only supplement caching with bound synthetic source and model evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from training import cache_unified_known_supplement as module


def fixture(monkeypatch, tmp_path, state_sha='b'*64):
    root = tmp_path/'sources'; root.mkdir()
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module, 'BASE_STATE_SHA', state_sha)
    bindings = {}
    for name in module.REQUIRED_SOURCES:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('Frozen '+name)
        bindings[str(path)] = module.sha(path)
    families = [f'Known {i}' for i in range(24)]+['__unknown__']
    for family, target in module.KNOWN_SOURCE_TARGETS.items():families[target] = family
    checkpoint = root/'base/model.pth'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'Exact frozen base')
    module.dump(checkpoint.parent/'SELECTION.json', {'families': families, 'state_after_sha256': state_sha})
    for path in (checkpoint, checkpoint.parent/'SELECTION.json'):bindings[str(path)] = module.sha(path)
    monkeypatch.setattr(module, 'BASE_CHECKPOINT_SHA', module.sha(checkpoint))
    monkeypatch.setattr(module, 'BASE_SELECTION_SHA', module.sha(checkpoint.parent/'SELECTION.json'))
    base_entry = module.entry(checkpoint)
    old_identity = {'architecture': module.ARCHITECTURE, 'base_checkpoint': base_entry, 'base_state_sha256': state_sha,
                    'families': families, 'bindings': dict(bindings), 'feature_device': 'cpu'}
    old = root/'old-cache'
    module.dump(old/'CACHE_FREEZE.json', {'schema': 'flux-glyph-retention-adapter-cache-freeze-v1',
        'identity': old_identity, 'optimizer_steps_executed': 0})
    module.dump(old/'CACHE_MANIFEST.json', {'schema': 'flux-glyph-retention-adapter-cache-v1',
        'identity': old_identity, 'cache_freeze_sha256': module.sha(old/'CACHE_FREEZE.json')})
    monkeypatch.setattr(module, 'OLD_CACHE_MANIFEST_SHA', module.sha(old/'CACHE_MANIFEST.json'))
    monkeypatch.setattr(module, 'OLD_CACHE_FREEZE_SHA', module.sha(old/'CACHE_FREEZE.json'))
    for path in (old/'CACHE_MANIFEST.json', old/'CACHE_FREEZE.json'):bindings[str(path)] = module.sha(path)
    data_root = root/'supplement'; (data_root/'train').mkdir(parents=True)
    proof = root/'native-font-proof.json'; proof.write_text('New real native font evidence')
    bindings[str(proof)] = module.sha(proof)
    preparation = {'schema': 'flux-glyph-unified-known-supplement-freeze-v1', 'families': families,
        'only_split': 'train', 'bindings': {str(proof): module.sha(proof)}}
    module.dump(data_root/'PREPARATION_FREEZE.json', preparation)
    manifest = {**preparation, 'schema': module.DATA_SCHEMA,
                'preparation_freeze_sha256': module.sha(data_root/'PREPARATION_FREEZE.json')}
    module.dump(data_root/'MANIFEST.json', manifest)
    tiles = np.random.default_rng(1407).random((4, 1, 64, 256), dtype=np.float32)
    tiles.tofile(data_root/'train/tiles.raw')
    rows = []
    for start, count, family in [(0, 1, 'LXGW WenKai'), (1, 2, 'WenQuanYi Micro Hei'), (3, 1, 'WenQuanYi Micro Hei')]:
        rows.append({'split': 'train', 'family': family, 'target': module.KNOWN_SOURCE_TARGETS[family], 'domain': 'android',
            'source_dataset': module.SOURCE_DATASET, 'source_id': 'known-'+str(start), 'region_id': 'known-region-'+str(start),
            'view': 'native', 'normalized_text_sha256': 'c'*64,
            'pair_evidence': {'source_id': 'unknown-'+str(start), 'region_id': 'unknown-region-'+str(start),
                'font_family': module.PAIRED_UNKNOWN_SOURCES[family], 'training_family': '__unknown__',
                'source_proof': str(proof), 'source_proof_sha256': module.sha(proof), 'source_image_sha256': 'd'*64,
                'normalized_text_sha256': 'c'*64, 'matched_fields': list(module.PAIR_FIELDS), 'intentional_train_text_pair': True},
            'native_font_verified': True, 'source_font_family': family, 'tile_start': start, 'tile_count': count,
            'tiles_sha256': hashlib.sha256(tiles[start:start+count].tobytes()).hexdigest()})
    module.dump(data_root/'train/rows.json', rows)
    part = {'schema': module.DATA_SCHEMA, 'split': 'train', 'families': families,
        'root_manifest_sha256': module.sha(data_root/'MANIFEST.json'), 'tiles': 4, 'views': 3,
        'shape': [4, 1, 64, 256],
        'metadata': {'path': 'rows.json', 'sha256': module.sha(data_root/'train/rows.json')},
        'array': {'path': 'tiles.raw', 'sha256': module.sha(data_root/'train/tiles.raw')}}
    module.dump(data_root/'train/MANIFEST.json', part)
    for path in [data_root/'MANIFEST.json', data_root/'PREPARATION_FREEZE.json',
                 *[data_root/'train'/name for name in ('MANIFEST.json', 'rows.json', 'tiles.raw')]]:
        bindings[str(path)] = module.sha(path)
    data = {'tiles': tiles, 'rows': rows, 'families': families, 'manifest': manifest, 'partition': part,
            'manifest_sha256': module.sha(data_root/'MANIFEST.json'),
            'partition_sha256': module.sha(data_root/'train/MANIFEST.json')}
    identity = module.cache_identity(data, base_entry, state_sha, bindings, 'cpu', data_root=data_root, old_cache=old)
    return data, identity, tmp_path/'cache'


def synthetic_cache(folder, identity):
    module.dump(folder/'CACHE_FREEZE.json', {'schema': module.FREEZE_SCHEMA, 'identity': identity,
        'feature_device': 'cpu', 'batch_size': 128, 'optimizer_steps_executed': 0})
    part = {}; (folder/'train').mkdir()
    for name, shape in [('features', (4, 128)), ('base_logits', (4, 25)), ('log_em_ratio', (4,))]:
        value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)/1000
        path = folder/'train'/(name+'.npy'); np.save(path, value)
        part[name] = {'path': 'train/'+name+'.npy', 'sha256': module.sha(path), 'shape': list(shape), 'dtype': 'float32'}
    manifest = {'schema': module.CACHE_SCHEMA, 'identity': identity,
        'cache_freeze_sha256': module.sha(folder/'CACHE_FREEZE.json'), 'partitions': {'train': part},
        'base_state_before_sha256': module.BASE_STATE_SHA, 'base_state_after_sha256': module.BASE_STATE_SHA,
        'optimizer_steps_executed': 0}
    module.dump(folder/'CACHE_MANIFEST.json', manifest)
    return manifest


def test_cache_loads_only_train_mmaps_and_preserves_the_original_source_order(monkeypatch, tmp_path):
    data, identity, folder = fixture(monkeypatch, tmp_path)
    expected = synthetic_cache(folder, identity)
    arrays, manifest = module.load_cache(folder)
    assert manifest == expected and set(arrays) == {'features', 'base_logits', 'log_em_ratio'}
    assert all(isinstance(v, np.memmap) and not v.flags.writeable for v in arrays.values())
    assert manifest['identity']['partition']['region_count'] == len(data['rows']) == 3
    assert manifest['identity']['partition']['tile_count'] == len(data['tiles']) == 4
    assert not (folder/'calibration').exists() and not (folder/'test').exists()
    assert module.load_cache(folder, identity)[1] == manifest
    assert identity['known_source_targets'] == {'LXGW WenKai': 10, 'WenQuanYi Micro Hei': 11}
    assert [row['target'] for row in data['rows']] == [10, 11, 11]
    assert all(row['family'] != row['pair_evidence']['training_family'] for row in data['rows'])


@pytest.mark.parametrize('fault', ['source_bytes', 'tile_bytes', 'rows_bytes', 'cache_bytes', 'extra_cal',
    'shape', 'dtype', 'nan', 'size_range', 'path_escape', 'base_changed', 'optimizer', 'unknown_order',
    'new_checkpoint', 'new_cache', 'missing_code', 'freeze_changed'])
def test_changed_sources_cache_or_execution_contract_is_rejected(monkeypatch, tmp_path, fault):
    _, identity, folder = fixture(monkeypatch, tmp_path); manifest = synthetic_cache(folder, identity)
    if fault in ('source_bytes', 'tile_bytes', 'rows_bytes'):
        path = {'source_bytes': module.ROOT/'native-font-proof.json',
                'tile_bytes': module.ROOT/'supplement/train/tiles.raw',
                'rows_bytes': module.ROOT/'supplement/train/rows.json'}[fault]
        path.write_bytes(b'Changed source')
    elif fault == 'cache_bytes':(folder/'train/features.npy').write_bytes(b'Changed feature bytes')
    elif fault == 'extra_cal':manifest['partitions']['calibration'] = manifest['partitions']['train']
    elif fault in ('shape', 'dtype', 'nan', 'size_range'):
        name = 'log_em_ratio' if fault == 'size_range' else 'features'
        value = np.zeros((4,) if name == 'log_em_ratio' else (4, 128), np.float32)
        if fault == 'shape':value = value[:3]
        elif fault == 'dtype':value = value.astype(np.float64)
        elif fault == 'nan':value[0, 0] = np.nan
        else:value[0] = 3.1
        path = folder/'train'/(name+'.npy'); np.save(path, value)
        manifest['partitions']['train'][name]['sha256'] = module.sha(path)
    elif fault == 'path_escape':manifest['partitions']['train']['features']['path'] = '../features.npy'
    elif fault == 'base_changed':manifest['base_state_after_sha256'] = 'a'*64
    elif fault == 'optimizer':manifest['optimizer_steps_executed'] = 1
    elif fault == 'unknown_order':manifest['identity']['families'].reverse()
    elif fault == 'new_checkpoint':monkeypatch.setattr(module, 'BASE_CHECKPOINT_SHA', 'a'*64)
    elif fault == 'new_cache':monkeypatch.setattr(module, 'OLD_CACHE_MANIFEST_SHA', 'a'*64)
    elif fault == 'missing_code':manifest['identity']['bindings'].pop(str(module.ROOT/module.REQUIRED_SOURCES[0]))
    else:
        freeze = module.read(folder/'CACHE_FREEZE.json'); freeze['optimizer_steps_executed'] = 1
        module.dump(folder/'CACHE_FREEZE.json', freeze)
        manifest['cache_freeze_sha256'] = module.sha(folder/'CACHE_FREEZE.json')
    module.dump(folder/'CACHE_MANIFEST.json', manifest)
    with pytest.raises(ValueError):module.load_cache(folder)


@pytest.mark.parametrize('fault', ['cal', 'unknown', 'target', 'domain', 'unverified', 'order', 'count', 'hash', 'nan', 'range'])
def test_only_verified_paired_known_train_pixels_in_exact_region_order_are_accepted(monkeypatch, tmp_path, fault):
    data, _, _ = fixture(monkeypatch, tmp_path)
    if fault == 'cal':data['rows'][0]['split'] = 'calibration'
    elif fault == 'unknown':data['rows'][0]['family'] = '__unknown__'
    elif fault == 'target':data['rows'][0]['target'] = 0
    elif fault == 'domain':data['rows'][0]['domain'] = 'ios'
    elif fault == 'unverified':data['rows'][0]['native_font_verified'] = False
    elif fault == 'order':data['rows'][1]['tile_start'] = 0
    elif fault == 'count':data['rows'][0]['tile_count'] = 9
    elif fault == 'hash':data['rows'][0]['tiles_sha256'] = 'a'*64
    elif fault == 'nan':data['tiles'][0, 0, 0, 0] = np.nan
    else:data['tiles'][0, 0, 0, 0] = 1.01
    with pytest.raises(ValueError):module.validate_data(data)


def test_real_cpu_extraction_matches_the_frozen_base_exactly_and_never_updates_weights(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(2)
    from retention_adapter_network import RetentionAdapterClassifier
    model = RetentionAdapterClassifier().requires_grad_(False).eval()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('residual_family_head.')}
    data, identity, folder = fixture(monkeypatch, tmp_path, module.state_sha(state))
    before = module.state_sha(model.state_dict())
    with torch.no_grad():
        features = model.features(torch.from_numpy(data['tiles']))
        logits = model.family_head(features); sizes = model.size_head(features).squeeze(-1)
    arrays, manifest = module.extract_cache(folder, model, data, identity, 'cpu')
    assert np.array_equal(arrays['features'], features.numpy())
    assert np.array_equal(arrays['base_logits'], logits.numpy())
    assert np.array_equal(arrays['log_em_ratio'], sizes.numpy())
    assert module.state_sha(model.state_dict()) == before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert manifest['optimizer_steps_executed'] == 0
    assert module.extract_cache(folder, model, data, identity, 'cpu')[1] == manifest
    model.family_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen eval'):module.extract_cache(tmp_path/'bad', model, data, identity, 'cpu')
    assert not (tmp_path/'bad').exists()


def test_cli_defaults_are_separate_supplement_data_cache_and_frozen_core():
    args = module.parser().parse_args([])
    assert args.data.name == 'data' and args.output.name == 'cache' and args.data.parent == args.output.parent
    assert args.output.parent.name == 'paired-known-capture-v1'
    assert args.checkpoint.parent.name == 'run-core-v1' and args.old_cache.parent.name == 'run-adapter-v1'
    assert args.device == 'mps'


@pytest.mark.parametrize('fault', ['dataset', 'pair_missing', 'paired_font', 'pair_target', 'pair_text',
    'pair_fields', 'pair_flag', 'pair_proof_binding', 'pair_source_self', 'swap_target', 'duplicate_view', 'only_one_family'])
def test_pair_provenance_cannot_relabel_known_targets_or_replace_the_matched_train_source(monkeypatch, tmp_path, fault):
    data, _, _ = fixture(monkeypatch, tmp_path)
    row = data['rows'][0]; pair = row['pair_evidence']
    if fault == 'dataset':row['source_dataset'] = 'android_new_unknown_supplement'
    elif fault == 'pair_missing':row.pop('pair_evidence')
    elif fault == 'paired_font':pair['font_family'] = 'Unbound Font'
    elif fault == 'pair_target':pair['training_family'] = row['family']
    elif fault == 'pair_text':pair['normalized_text_sha256'] = 'f'*64
    elif fault == 'pair_fields':pair['matched_fields'].remove('font_size_px')
    elif fault == 'pair_flag':pair['intentional_train_text_pair'] = False
    elif fault == 'pair_proof_binding':pair['source_proof_sha256'] = 'f'*64
    elif fault == 'pair_source_self':pair['source_id'] = row['source_id']
    elif fault == 'swap_target':row['target'] = 11
    elif fault == 'duplicate_view':
        for key in ('source_id', 'region_id', 'view'):data['rows'][1][key] = row[key]
    else:
        for item in data['rows']:
            item['family'] = item['source_font_family'] = 'LXGW WenKai'; item['target'] = 10
            item['pair_evidence']['font_family'] = 'Zhuque Fangsong'
    with pytest.raises(ValueError):module.validate_data(data)


def test_loader_rechecks_known_targets_even_when_changed_rows_and_cache_are_rehashed(monkeypatch, tmp_path):
    _, identity, folder = fixture(monkeypatch, tmp_path)
    manifest = synthetic_cache(folder, identity)
    part_path = Path(identity['partition_manifest']['path']); part = module.read(part_path)
    rows_path = part_path.parent/part['metadata']['path']; rows = module.read(rows_path)
    rows[0]['family'] = '__unknown__'; rows[0]['target'] = 24
    module.dump(rows_path, rows)
    part['metadata']['sha256'] = module.sha(rows_path); module.dump(part_path, part)
    identity['bindings'][str(rows_path)] = module.sha(rows_path)
    identity['bindings'][str(part_path)] = module.sha(part_path)
    identity['partition']['rows'] = part['metadata']
    identity['partition_manifest']['sha256'] = module.sha(part_path)
    freeze = module.read(folder/'CACHE_FREEZE.json'); freeze['identity'] = identity
    module.dump(folder/'CACHE_FREEZE.json', freeze)
    manifest['identity'] = identity; manifest['cache_freeze_sha256'] = module.sha(folder/'CACHE_FREEZE.json')
    module.dump(folder/'CACHE_MANIFEST.json', manifest)
    with pytest.raises(ValueError, match='unchanged labels'):module.load_cache(folder)
