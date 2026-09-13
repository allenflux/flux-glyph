#!/usr/bin/env python3
"""Extract an independently bound, TRAIN-only cache of new paired real known fonts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import dump, require, sha, state_sha

ARCHITECTURE = 'region-cnn64x256-residual-head-v1'
DATA_SCHEMA = 'flux-glyph-unified-known-supplement-v1'
CACHE_SCHEMA = 'flux-glyph-unified-known-supplement-cache-v1'
FREEZE_SCHEMA = 'flux-glyph-unified-known-supplement-cache-freeze-v1'
BASE_CHECKPOINT_SHA = 'f2f7f7c6d44aad5b3e22a2ba7d5c78b069fbb0a04423ee9bf186bfa60e52533e'
BASE_SELECTION_SHA = '18b5cbf95e45fee83283256581f32c94dae80f653e2fdfe1e24ddf25bfb414da'
BASE_STATE_SHA = 'cb2a9bbe2e7e579d5d3a49ff7126c62f481ca94046d90ba250bbc4fa16c14bd8'
OLD_CACHE_MANIFEST_SHA = 'c3d3e57cf4d8377174e4b7981bbc1d30ee0e72af547f41f2f96a9a674998974a'
OLD_CACHE_FREEZE_SHA = 'b9e4ea89297911283610702afc67ecf31bc70b724258b1399e01a364e5180bba'
BATCH_SIZE = 128
SOURCE_DATASET = 'android_paired_known_supplement'
KNOWN_SOURCE_TARGETS = {'LXGW WenKai': 10, 'WenQuanYi Micro Hei': 11}
PAIR_FIELDS = ['text', 'script', 'font_size_px', 'color', 'bbox', 'page_background', 'canvas_px']
PAIRED_UNKNOWN_SOURCES = {'LXGW WenKai': 'Zhuque Fangsong', 'WenQuanYi Micro Hei': 'WenQuanYi Zen Hei'}
ORDER = 'Exact prepared TRAIN tile order; source rows retain tile_start and tile_count.'
REQUIRED_SOURCES = ('training/cache_unified_known_supplement.py', 'training/prepare_unified_known_supplement.py',
    'training/retention_adapter_network.py', 'training/region_network.py', 'training/network.py', 'training/train_regions.py')


def read(path):
    return json.loads(Path(path).read_text())


def entry(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verify_bindings(bindings):
    require(isinstance(bindings, dict) and bindings, 'Missing supplement cache source bindings')
    for name, digest in bindings.items():
        require(isinstance(name, str) and Path(name).is_absolute() and Path(name).is_file()
                and isinstance(digest, str) and sha(name) == digest, 'Supplement cache source changed: ' + str(name))


def validate_rows(rows, families, tile_count, bindings):
    """Known targets stay known; pair evidence describes a different TRAIN source."""
    require(isinstance(rows, list) and rows and len(families) == len(set(families)) == 25
            and all(families[target] == family for family, target in KNOWN_SOURCE_TARGETS.items()),
            'Paired known source classes or registry indices differ')
    offset = 0; seen = set(); found = set()
    for row in rows:
        family = row.get('family'); count = row.get('tile_count')
        require(family in KNOWN_SOURCE_TARGETS and row.get('source_font_family') == family
                and type(row.get('target')) is int and row['target'] == KNOWN_SOURCE_TARGETS[family]
                and row.get('split') == 'train' and row.get('domain') == 'android'
                and row.get('source_dataset') == SOURCE_DATASET and row.get('native_font_verified') is True
                and type(row.get('tile_start')) is int and row['tile_start'] == offset
                and type(count) is int and 1 <= count <= 8 and offset+count <= tile_count,
                'Cache requires verified paired known TRAIN rows with unchanged labels and tile order')
        key = (row.get('source_id'), row.get('region_id'), row.get('view'))
        require(all(isinstance(value, str) and value for value in key) and key not in seen
                and key[2] in ('native', 'half', 'three_quarters_jpeg75', 'jpeg75'),
                'Invalid or duplicate paired known TRAIN region/view identity')
        pair = row.get('pair_evidence')
        require(isinstance(pair, dict) and pair.get('font_family') == PAIRED_UNKNOWN_SOURCES[family]
                and pair.get('training_family') == '__unknown__' and pair.get('intentional_train_text_pair') is True
                and pair.get('matched_fields') == PAIR_FIELDS
                and all(isinstance(pair.get(field), str) and pair[field] for field in ('source_id', 'region_id', 'source_proof'))
                and pair['source_id'] != key[0]
                and Path(pair['source_proof']).is_absolute()
                and bindings.get(pair['source_proof']) == pair.get('source_proof_sha256')
                and pair.get('normalized_text_sha256') == row.get('normalized_text_sha256')
                and all(isinstance(pair.get(field), str) and len(pair[field]) == 64
                        and all(char in '0123456789abcdef' for char in pair[field])
                        for field in ('source_proof_sha256', 'source_image_sha256', 'normalized_text_sha256')),
                'Paired TRAIN proof, source mapping or matched content identity differs')
        seen.add(key); found.add(family); offset += count
    require(offset == tile_count and found == set(KNOWN_SOURCE_TARGETS),
            'Paired known TRAIN cache must cover both real known fonts and all tiles')


def validate_data(data):
    """Check only the supplied new TRAIN array, row order and per-region bytes."""
    tiles, rows, families, part = data['tiles'], data['rows'], data['families'], data['partition']
    require(data['manifest']['schema'] == part['schema'] == DATA_SCHEMA and part['split'] == 'train'
            and isinstance(families, list) and len(families) == len(set(families)) == 25
            and families[-1] == '__unknown__' and families.count('__unknown__') == 1
            and part['families'] == data['manifest']['families'] == families
            and part['root_manifest_sha256'] == data['manifest_sha256']
            and isinstance(tiles, np.ndarray) and tiles.dtype == np.float32
            and tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256) and len(tiles) > 0
            and part['shape'] == list(tiles.shape) and part['tiles'] == len(tiles)
            and isinstance(rows, list) and rows and part['views'] == len(rows),
            'Invalid TRAIN supplement array or class identity')
    validate_rows(rows, families, len(tiles), data['manifest']['bindings'])
    offset = 0
    for row in rows:
        count = row.get('tile_count')
        block = tiles[offset:offset + count]
        require(bool(np.isfinite(block).all()) and bool(((block >= 0) & (block <= 1)).all())
                and hashlib.sha256(block.tobytes()).hexdigest() == row.get('tiles_sha256'),
                'Supplement TRAIN tile bytes or pixel range differ')
        offset += count
    require(offset == len(tiles), 'Unclaimed supplement TRAIN tiles')


def cache_identity(data, base_checkpoint, base_state_sha256, bindings, feature_device, *, data_root, old_cache):
    validate_data(data)
    data_root, old_cache = Path(data_root).resolve(), Path(old_cache).resolve()
    base_selection = Path(base_checkpoint['path']).resolve().parent/'SELECTION.json'
    identity = {'architecture': ARCHITECTURE, 'data_schema': DATA_SCHEMA, 'split': 'train',
        'source_dataset': SOURCE_DATASET, 'known_source_targets': dict(KNOWN_SOURCE_TARGETS),
        'paired_unknown_sources': dict(PAIRED_UNKNOWN_SOURCES),
        'feature_device': feature_device, 'families': list(data['families']),
        'base_checkpoint': dict(base_checkpoint), 'base_selection': entry(base_selection),
        'base_state_sha256': base_state_sha256,
        'old_cache_manifest': entry(old_cache/'CACHE_MANIFEST.json'),
        'old_cache_freeze': entry(old_cache/'CACHE_FREEZE.json'),
        'data_manifest': entry(data_root/'MANIFEST.json'),
        'preparation_freeze': entry(data_root/'PREPARATION_FREEZE.json'),
        'partition_manifest': entry(data_root/'train/MANIFEST.json'),
        'partition': {'split': 'train', 'tile_count': len(data['tiles']), 'region_count': len(data['rows']),
            'rows': dict(data['partition']['metadata']), 'tiles': dict(data['partition']['array']), 'order': ORDER},
        'bindings': dict(bindings), 'optimizer_steps_executed': 0, 'test_read': False,
        'development_holdout_read': False, 'calibration_images_read': False,
        'outputs': 'Frozen base features, family_head logits and size_head only; no residual correction.'}
    require(identity['data_manifest']['sha256'] == data['manifest_sha256']
            and identity['partition_manifest']['sha256'] == data['partition_sha256'],
            'Supplement data objects differ from their bound manifests')
    validate_identity(identity)
    return identity


def validate_identity(identity):
    require(identity.get('architecture') == ARCHITECTURE and identity.get('data_schema') == DATA_SCHEMA
            and identity.get('source_dataset') == SOURCE_DATASET
            and identity.get('known_source_targets') == KNOWN_SOURCE_TARGETS
            and identity.get('paired_unknown_sources') == PAIRED_UNKNOWN_SOURCES
            and identity.get('split') == 'train' and identity.get('feature_device') in ('cpu', 'mps')
            and identity.get('optimizer_steps_executed') == 0 and identity.get('test_read') is False
            and identity.get('development_holdout_read') is False and identity.get('calibration_images_read') is False
            and identity.get('base_state_sha256') == BASE_STATE_SHA
            and identity.get('outputs') == 'Frozen base features, family_head logits and size_head only; no residual correction.',
            'Supplement cache execution or frozen model contract differs')
    bindings = identity['bindings']; verify_bindings(bindings)
    require(all(bindings.get(str((ROOT/name).resolve())) == sha(ROOT/name) for name in REQUIRED_SOURCES),
            'Missing supplement preparation, cache or network source binding')
    for name in ('base_checkpoint', 'base_selection', 'old_cache_manifest', 'old_cache_freeze',
                 'data_manifest', 'preparation_freeze', 'partition_manifest'):
        item = identity[name]
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute()
                and bindings.get(item['path']) == item['sha256'], 'Unbound supplement cache identity: ' + name)
    require(identity['base_checkpoint']['sha256'] == BASE_CHECKPOINT_SHA
            and identity['base_selection']['sha256'] == BASE_SELECTION_SHA
            and Path(identity['base_selection']['path']) == Path(identity['base_checkpoint']['path']).parent/'SELECTION.json'
            and identity['old_cache_manifest']['sha256'] == OLD_CACHE_MANIFEST_SHA
            and identity['old_cache_freeze']['sha256'] == OLD_CACHE_FREEZE_SHA
            and Path(identity['old_cache_freeze']['path']) == Path(identity['old_cache_manifest']['path']).parent/'CACHE_FREEZE.json',
            'Supplement cache must use the exact core step-1000 base and existing feature cache')
    old = read(identity['old_cache_manifest']['path']); old_freeze = read(identity['old_cache_freeze']['path'])
    require(old.get('schema') == 'flux-glyph-retention-adapter-cache-v1'
            and old.get('cache_freeze_sha256') == OLD_CACHE_FREEZE_SHA
            and old_freeze.get('schema') == 'flux-glyph-retention-adapter-cache-freeze-v1'
            and old_freeze.get('identity') == old['identity']
            and old['identity']['architecture'] == ARCHITECTURE
            and old['identity']['base_checkpoint'] == identity['base_checkpoint']
            and old['identity']['base_state_sha256'] == BASE_STATE_SHA
            and old['identity']['families'] == identity['families']
            and all(bindings.get(path) == digest for path, digest in old['identity']['bindings'].items()),
            'Original feature cache model, families or source closure differs')
    manifest = read(identity['data_manifest']['path']); part = read(identity['partition_manifest']['path'])
    preparation = read(identity['preparation_freeze']['path'])
    data_root = Path(identity['data_manifest']['path']).parent
    require(Path(identity['partition_manifest']['path']) == data_root/'train/MANIFEST.json'
            and Path(identity['preparation_freeze']['path']) == data_root/'PREPARATION_FREEZE.json'
            and manifest.get('schema') == part.get('schema') == DATA_SCHEMA and part.get('split') == 'train'
            and manifest.get('only_split') == 'train'
            and preparation.get('schema') == 'flux-glyph-unified-known-supplement-freeze-v1'
            and manifest.get('preparation_freeze_sha256') == identity['preparation_freeze']['sha256']
            and all(manifest.get(key) == value for key, value in preparation.items() if key != 'schema')
            and manifest['families'] == part['families'] == identity['families']
            and len(identity['families']) == len(set(identity['families'])) == 25
            and identity['families'][-1] == '__unknown__'
            and part['root_manifest_sha256'] == identity['data_manifest']['sha256']
            and all(bindings.get(path) == digest for path, digest in manifest['bindings'].items()),
            'Prepared supplement source identity differs')
    require(identity['partition'] == {'split': 'train', 'tile_count': part['tiles'], 'region_count': part['views'],
            'rows': part['metadata'], 'tiles': part['array'], 'order': ORDER}
            and type(part['tiles']) is int and part['tiles'] > 0 and type(part['views']) is int and part['views'] > 0
            and part['shape'] == [part['tiles'], 1, 64, 256], 'Prepared TRAIN partition geometry differs')
    for name in ('metadata', 'array'):
        item = part[name]; path = (data_root/'train'/item['path']).resolve()
        require(path.parent == data_root/'train' and bindings.get(str(path)) == item['sha256'],
                'Unbound or escaping supplement TRAIN asset')
    rows = read(data_root/'train'/part['metadata']['path'])
    require(len(rows) == part['views'], 'Paired known TRAIN row count differs')
    validate_rows(rows, identity['families'], part['tiles'], bindings)


def load_cache(folder, identity=None):
    """Load only new TRAIN mmap arrays; caller should also bind CACHE_MANIFEST SHA."""
    folder = Path(folder).resolve()
    manifest = read(folder/'CACHE_MANIFEST.json'); freeze = read(folder/'CACHE_FREEZE.json')
    expected = manifest['identity'] if identity is None else identity
    validate_identity(expected)
    require(manifest.get('schema') == CACHE_SCHEMA and manifest.get('identity') == expected
            and manifest.get('cache_freeze_sha256') == sha(folder/'CACHE_FREEZE.json')
            and freeze.get('schema') == FREEZE_SCHEMA and freeze.get('identity') == expected
            and freeze.get('feature_device') == expected['feature_device'] and freeze.get('batch_size') == BATCH_SIZE
            and freeze.get('optimizer_steps_executed') == 0
            and manifest.get('base_state_before_sha256') == manifest.get('base_state_after_sha256') == BASE_STATE_SHA
            and manifest.get('optimizer_steps_executed') == 0 and set(manifest['partitions']) == {'train'},
            'Frozen TRAIN-only supplement cache contract differs')
    count = expected['partition']['tile_count']; part = manifest['partitions']['train']; arrays = {}
    require(set(part) == {'features', 'base_logits', 'log_em_ratio'}, 'Supplement cache outputs differ')
    for name, shape in (('features', (count, 128)), ('base_logits', (count, 25)), ('log_em_ratio', (count,))):
        item = part[name]; relative = 'train/'+name+'.npy'; path = folder/relative
        require(item.get('path') == relative and item.get('shape') == list(shape) and item.get('dtype') == 'float32'
                and sha(path) == item.get('sha256'), 'Supplement cached array binding differs')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == shape and bool(np.isfinite(array).all()),
                'Invalid supplement feature or head output')
        if name == 'log_em_ratio':require(bool((np.abs(array) <= 3).all()), 'Invalid frozen supplement size output')
        arrays[name] = array
    return arrays, manifest


def extract_cache(folder, model, data, identity, device):
    import torch
    folder = Path(folder).resolve()
    validate_data(data); validate_identity(identity)
    require(device == identity['feature_device'] and data['manifest_sha256'] == identity['data_manifest']['sha256']
            and data['partition_sha256'] == identity['partition_manifest']['sha256']
            and data['families'] == identity['families'], 'Extraction data or device differs from frozen identity')
    require(not model.training and all(not p.requires_grad and p.grad is None and p.dtype == torch.float32
            for p in model.parameters()), 'Cache extraction requires a frozen eval float32 base')
    base = {key: value for key, value in model.state_dict().items() if not key.startswith('residual_family_head.')}
    before = state_sha(base); full_before = state_sha(model.state_dict())
    require(before == identity['base_state_sha256'], 'Cache model state differs from the frozen base')
    if folder.exists():return load_cache(folder, identity)
    folder.mkdir(parents=True)
    dump(folder/'CACHE_FREEZE.json', {'schema': FREEZE_SCHEMA, 'identity': identity, 'feature_device': device,
        'torch_version': str(torch.__version__), 'batch_size': BATCH_SIZE, 'optimizer_steps_executed': 0})
    freeze_sha = sha(folder/'CACHE_FREEZE.json'); directory = folder/'train'; directory.mkdir()
    count = len(data['tiles']); model.to(device)
    arrays = {name: np.lib.format.open_memmap(directory/(name+'.npy'), mode='w+', dtype=np.float32, shape=shape)
        for name, shape in (('features', (count, 128)), ('base_logits', (count, 25)), ('log_em_ratio', (count,)))}
    with torch.inference_mode():
        for start in range(0, count, BATCH_SIZE):
            tiles = torch.from_numpy(np.array(data['tiles'][start:start + BATCH_SIZE], copy=True)).to(device)
            features = model.features(tiles)
            outputs = {'features': features, 'base_logits': model.family_head(features),
                       'log_em_ratio': model.size_head(features).squeeze(-1)}
            for name, value in outputs.items():
                array = value.cpu().numpy()
                require(array.dtype == np.float32 and array.shape == arrays[name][start:start + len(tiles)].shape
                        and bool(np.isfinite(array).all()), 'Invalid extracted supplement output')
                if name == 'log_em_ratio':require(bool((np.abs(array) <= 3).all()), 'Invalid extracted frozen size output')
                arrays[name][start:start + len(tiles)] = array
    part = {}
    for name, array in arrays.items():
        array.flush(); path = directory/(name+'.npy')
        part[name] = {'path': 'train/'+name+'.npy', 'sha256': sha(path), 'shape': list(array.shape), 'dtype': 'float32'}
    after = state_sha({key: value for key, value in model.state_dict().items() if not key.startswith('residual_family_head.')})
    require(before == after and state_sha(model.state_dict()) == full_before
            and sha(folder/'CACHE_FREEZE.json') == freeze_sha
            and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'Feature extraction altered model parameters or its freeze')
    validate_data(data); validate_identity(identity)
    dump(folder/'CACHE_MANIFEST.json', {'schema': CACHE_SCHEMA, 'identity': identity,
        'cache_freeze_sha256': freeze_sha, 'partitions': {'train': part},
        'base_state_before_sha256': before, 'base_state_after_sha256': after, 'optimizer_steps_executed': 0})
    return load_cache(folder, identity)


def prepare(args):
    import torch
    from prepare_unified_known_supplement import load_supplement
    from retention_adapter_network import RetentionAdapterClassifier
    args.data, args.output, args.checkpoint, args.old_cache = (
        Path(value).resolve() for value in (args.data, args.output, args.checkpoint, args.old_cache))
    require(args.output != args.old_cache and args.old_cache not in args.output.parents
            and args.data != args.output and args.data not in args.output.parents, 'Supplement cache needs a separate new directory')
    require(sha(args.checkpoint) == BASE_CHECKPOINT_SHA and sha(args.checkpoint.parent/'SELECTION.json') == BASE_SELECTION_SHA
            and sha(args.old_cache/'CACHE_MANIFEST.json') == OLD_CACHE_MANIFEST_SHA
            and sha(args.old_cache/'CACHE_FREEZE.json') == OLD_CACHE_FREEZE_SHA, 'Frozen cache or core initializer differs')
    data = load_supplement(args.data); old = read(args.old_cache/'CACHE_MANIFEST.json')
    selection = read(args.checkpoint.parent/'SELECTION.json')
    require(selection['selected']['step'] == 1000 and selection['optimizer_steps_executed'] == 1500
            and selection['state_after_sha256'] == BASE_STATE_SHA and selection['families'] == data['families']
            and selection['test_read'] is False and selection['development_holdout_read'] is False,
            'Supplement source checkpoint lineage differs')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    require(checkpoint.get('architecture') == 'region-cnn64x256-unified-v1'
            and checkpoint.get('selection_sha256') == BASE_SELECTION_SHA and checkpoint.get('families') == data['families']
            and state_sha(checkpoint['state_dict']) == BASE_STATE_SHA, 'Supplement initializer state differs')
    model = RetentionAdapterClassifier(25).from_parent(checkpoint['state_dict']).requires_grad_(False).eval()
    bindings = dict(old['identity']['bindings'])
    for path, digest in data['manifest']['bindings'].items():
        require(path not in bindings or bindings[path] == digest, 'Old cache and supplement provenance conflict')
        bindings[path] = digest
    paths = [Path(__file__), ROOT/'training/prepare_unified_known_supplement.py',
        ROOT/'training/retention_adapter_network.py', ROOT/'training/region_network.py', ROOT/'training/network.py',
        ROOT/'training/train_regions.py', args.checkpoint, args.checkpoint.parent/'SELECTION.json',
        args.old_cache/'CACHE_MANIFEST.json', args.old_cache/'CACHE_FREEZE.json',
        args.data/'MANIFEST.json', args.data/'PREPARATION_FREEZE.json', args.data/'train/MANIFEST.json']
    paths += [args.data/'train'/data['partition'][name]['path'] for name in ('array', 'metadata')]
    for path in paths:
        bound = entry(path)
        require(bound['path'] not in bindings or bindings[bound['path']] == bound['sha256'], 'Supplement source binding conflict')
        bindings[bound['path']] = bound['sha256']
    identity = cache_identity(data, entry(args.checkpoint), BASE_STATE_SHA, bindings, args.device,
                              data_root=args.data, old_cache=args.old_cache)
    torch.set_num_threads(4)
    _, manifest = extract_cache(args.output, model, data, identity, args.device)
    print(json.dumps({'schema': CACHE_SCHEMA, 'rows': identity['partition']['region_count'],
        'tiles': identity['partition']['tile_count'], 'device': args.device, 'optimizer_steps_executed': 0,
        'cache_manifest_sha256': sha(args.output/'CACHE_MANIFEST.json'),
        'base_unchanged': manifest['base_state_before_sha256'] == manifest['base_state_after_sha256'],
        'test_read': False, 'development_holdout_read': False}, ensure_ascii=False), flush=True)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'artifacts/unified-font-v2/paired-known-capture-v1'
    parser.add_argument('--data', type=Path, default=base/'data')
    parser.add_argument('--output', type=Path, default=base/'cache')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/unified-font-v2/run-core-v1/model.pth')
    parser.add_argument('--old-cache', type=Path, default=ROOT/'artifacts/unified-font-v2/run-adapter-v1/cache')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    return parser


if __name__ == '__main__':
    prepare(parser().parse_args())
