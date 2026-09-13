#!/usr/bin/env python3
"""Recompute one pinned student's logits for three TRAIN partitions, with no optimizer.

Only prepared TRAIN pixels are opened. Historical preparation/source closures are
bound by their immutable manifest bytes; their non-TRAIN arrays are not reopened.
"""
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
from train_unified_retention import parameter_groups, retention_rank
from prepare_unified_regions import FAMILIES

ARCHITECTURE = 'region-cnn64x256-unified-v1'
CACHE_SCHEMA = 'flux-glyph-unified-student-teacher-cache-v1'
FREEZE_SCHEMA = 'flux-glyph-unified-student-teacher-cache-freeze-v1'
TEACHER_CHECKPOINT_SHA = 'b08c0e20e98835e55094debedcad495d579d478b8f793aa733601383bec87763'
TEACHER_SELECTION_SHA = '14825ee2c994aeb1a387f590f95ff4354efe6323ab92891230a47c69837002d8'
TEACHER_FREEZE_SHA = '3722b7a9d940fc069718e32c98bcb437dc01601470e5778160b3271fefc19d28'
TEACHER_STATE_SHA = 'ab820e0ba8672107d3a5aecc69a3dc6418a0496ced78c0e4f5e898f30db9435c'
BATCH_SIZE = 128
PARTITIONS = ('original', 'supplement', 'known')
DATA_ROOTS = {
    'original': ROOT/'artifacts/unified-font-v1/data-v1',
    'supplement': ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/data',
    'known': ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/data'}
# Immutable prepared datasets, independently audited before this extraction.
DATA_PINS = {
    'original': ('dfdd9fa746b28b3665eed10491ce83c5c872f16b2499fe3ae6c6e6bb9ed40a23',
        '4a14ef83b39181f5da55d0c5a7c3a0f868547e6dd5ee497f5887b70cec479ca5', 70128, 130854),
    'supplement': ('43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05',
        'df2c79fc8f99a4bd364e631a97e39a3c8d0281cf68a01199316143898664425c', 1918, 3156),
    'known': ('4caff363a62c4df7f58dd3f63fe271dff7c0e6a91952cd5bff7958e78d60e357',
        'c5e5076e659c81b5ca182b08fa7d7beb05fb1cc4aff418445ce7144d9c15620a', 1920, 3220)}
DATA_SCHEMAS = {'original': 'flux-glyph-unified-region-training-data-v1',
    'supplement': 'flux-glyph-unified-unknown-supplement-v1',
    'known': 'flux-glyph-unified-known-supplement-v1'}
REQUIRED_SOURCES = ('training/cache_unified_student_teacher.py', 'training/region_network.py',
    'training/network.py', 'training/train_regions.py', 'training/train_unified_retention.py',
    'training/prepare_unified_regions.py', 'training/prepare_unified_unknown_supplement.py',
    'training/prepare_unified_known_supplement.py')
ORDER = 'Exact prepared TRAIN tile_start order; no resampling, shuffling or region aggregation.'
SCOPE = 'Prepared TRAIN bytes verified; historical closure manifests and code verified without reopening non-TRAIN arrays.'


def read(path):
    return json.loads(Path(path).read_text())


def entry(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def bound_entry(item, bindings):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256'}
        and isinstance(item['path'], str) and Path(item['path']).is_absolute()
        and bindings.get(item['path']) == item['sha256'], 'Unbound teacher cache identity')


def teacher_identity(checkpoint):
    path = Path(checkpoint).resolve(); folder = path.parent
    items = {name: entry(folder/file) for name, file in
        (('checkpoint', 'model.pth'), ('selection', 'SELECTION.json'), ('training_freeze', 'TRAINING_FREEZE.json'))}
    require(path == folder/'model.pth' and items['checkpoint']['sha256'] == TEACHER_CHECKPOINT_SHA
        and items['selection']['sha256'] == TEACHER_SELECTION_SHA
        and items['training_freeze']['sha256'] == TEACHER_FREEZE_SHA,
        'Teacher must be the exact source-balanced b08 selected checkpoint, not core or an average')
    selection = read(items['selection']['path']); freeze = read(items['training_freeze']['path'])
    require(selection.get('schema') == 'flux-glyph-unified-retention-source-balanced-selection-v1'
        and freeze.get('schema') == 'flux-glyph-unified-retention-source-balanced-protocol-v1'
        and selection.get('training_protocol_sha256') == TEACHER_FREEZE_SHA
        and all(selection.get(k) == v for k, v in freeze.items() if k != 'schema')
        and selection.get('architecture') == ARCHITECTURE and selection.get('families') == list(FAMILIES)
        and selection.get('optimizer_steps_executed') == 3000
        and [r['step'] for r in selection['history']] == list(range(500, 3001, 500))
        and selection['selected'] == max(selection['history'], key=retention_rank)
        and selection['selected']['step'] == 1500
        and selection.get('state_after_sha256') == TEACHER_STATE_SHA
        and selection.get('test_read') is False and selection.get('development_holdout_read') is False,
        'Teacher source completion, original rank or full-state identity differs')
    groups = selection['selected_parameter_groups_sha256']
    require(set(groups) == {'trunk', 'style', 'family_head', 'size_head'}
        and all(groups[k] != selection['initial_parameter_groups_sha256'][k]
            and selection['final_parameter_groups_sha256'][k] != selection['initial_parameter_groups_sha256'][k]
            for k in groups), 'Teacher source did not update all four original parameter groups')
    return {**items, 'state_sha256': TEACHER_STATE_SHA, 'parameter_groups_sha256': groups,
            'selected_step': 1500, 'optimizer_steps_executed': 3000}, selection


def partition_identity(name, root):
    """Resolve exactly four prepared files, plus a TRAIN supplement freeze."""
    require(name in PARTITIONS, 'Unknown teacher source partition')
    root = Path(root).resolve(); manifest_item = entry(root/'MANIFEST.json')
    part_item = entry(root/'train/MANIFEST.json'); manifest = read(manifest_item['path']); part = read(part_item['path'])
    pins = DATA_PINS[name]
    require((manifest_item['sha256'], part_item['sha256'], part.get('views'), part.get('tiles')) == pins
        and manifest.get('schema') == part.get('schema') == DATA_SCHEMAS[name]
        and manifest.get('families') == part.get('families') == list(FAMILIES)
        and part.get('root_manifest_sha256') == manifest_item['sha256'] and part.get('split') == 'train'
        and part.get('shape') == [pins[3], 1, 64, 256], 'Teacher source must match the pinned TRAIN data')
    assets = {}
    for key, expected in (('metadata', 'rows.json'), ('array', 'tiles.raw')):
        item = part[key]; path = (root/'train'/expected).resolve()
        require(item.get('path') == expected and path.parent == root/'train', 'Escaping TRAIN asset path')
        assets[key] = entry(path)
        require(assets[key]['sha256'] == item['sha256'], 'Prepared TRAIN asset bytes changed')
    require(Path(assets['array']['path']).stat().st_size == pins[3]*64*256*4, 'TRAIN array byte length differs')
    result = {'split': 'train', 'data_manifest': manifest_item, 'partition_manifest': part_item,
        'rows': assets['metadata'], 'tiles': assets['array'], 'region_count': pins[2], 'tile_count': pins[3],
        'shape': [pins[3], 1, 64, 256], 'order': ORDER}
    if name != 'original':
        result['preparation_freeze'] = entry(root/'PREPARATION_FREEZE.json')
        preparation = read(result['preparation_freeze']['path'])
        require(manifest.get('only_split') == 'train'
            and manifest.get('preparation_freeze_sha256') == result['preparation_freeze']['sha256']
            and all(manifest.get(k) == v for k, v in preparation.items() if k != 'schema'),
            'Supplement preparation freeze changed')
    return result, manifest


def source_bindings(teacher, selection, partitions, manifests):
    """Explicit allowlist: no inherited raw/CAL pixel bindings are opened."""
    result = {}
    for document in (teacher, *partitions.values()):
        for item in document.values():
            if isinstance(item, dict) and set(item) == {'path', 'sha256'}:
                result[item['path']] = item['sha256']
    for document in (selection, *manifests.values()):
        for path, digest in document.get('bindings', {}).items():
            if Path(path).suffix == '.py':
                require(Path(path).is_absolute() and sha(path) == digest, 'Historical preparation or model source changed')
                require(path not in result or result[path] == digest, 'Contradictory teacher source closure')
                result[path] = digest
    for relative in REQUIRED_SOURCES:
        item = entry(ROOT/relative)
        require(item['path'] not in result or result[item['path']] == item['sha256'], 'Frozen cache dependency changed')
        result[item['path']] = item['sha256']
    return result


def cache_identity(checkpoint, roots, device):
    require(set(roots) == set(PARTITIONS) and device in ('cpu', 'mps'), 'Teacher cache sources or device differ')
    teacher, selection = teacher_identity(checkpoint); partitions = {}; manifests = {}
    for name in PARTITIONS:
        partitions[name], manifests[name] = partition_identity(name, roots[name])
    return {'architecture': ARCHITECTURE, 'families': list(FAMILIES), 'teacher': teacher,
        'partitions': partitions, 'bindings': source_bindings(teacher, selection, partitions, manifests),
        'inference_device': device, 'batch_size': BATCH_SIZE, 'optimizer_steps_executed': 0,
        'teacher_optimizer_steps_executed': 0, 'teacher_model_count': 1, 'outputs': ['base_logits'],
        'teacher_logits_recomputed': True, 'historical_logits_reused': False,
        'test_read': False, 'development_holdout_read': False, 'calibration_images_read': False,
        'verification_scope': SCOPE, 'total_tiles': sum(p['tile_count'] for p in partitions.values())}


def validate_identity(identity):
    require(isinstance(identity, dict) and set(identity.get('partitions', {})) == set(PARTITIONS),
        'Teacher cache must contain exactly the three TRAIN partitions')
    roots = {name: Path(identity['partitions'][name]['data_manifest']['path']).parent for name in PARTITIONS}
    expected = cache_identity(identity['teacher']['checkpoint']['path'], roots, identity['inference_device'])
    require(identity == expected, 'Teacher cache source, model, scope or row-order identity changed')
    return expected


def validate_rows(rows, families, count, name):
    require(isinstance(rows, list) and rows and families == list(FAMILIES), 'Invalid TRAIN class or row registry')
    offset = 0; seen = set()
    for row in rows:
        target = row.get('target'); n = row.get('tile_count')
        key = (row.get('source_id'), row.get('region_id'), row.get('view'))
        require(row.get('split') == row.get('original_split') == 'train'
            and row.get('native_font_verified') is True and row.get('domain') in ('ios', 'android')
            and type(target) is int and 0 <= target < 25 and row.get('family') == families[target]
            and all(isinstance(x, str) and x for x in key) and key not in seen
            and key[2] in ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
            and type(row.get('tile_start')) is int and row['tile_start'] == offset
            and type(n) is int and 1 <= n <= 8 and offset+n <= count,
            'Invalid TRAIN label, verification, split, view or tile order')
        if name == 'supplement':
            require(target == 24 and row.get('source_dataset') == 'android_new_unknown_supplement'
                and row.get('source_font_family') in ('WenQuanYi Zen Hei', 'Zhuque Fangsong')
                and row['domain'] == 'android', 'Invalid unknown supplement TRAIN source')
        elif name == 'known':
            require(target in (10, 11) and row.get('source_dataset') == 'android_paired_known_supplement'
                and row.get('source_font_family') == families[target] and row['domain'] == 'android',
                'Invalid paired known TRAIN source')
        seen.add(key); offset += n
    require(offset == count, 'Unclaimed TRAIN tiles')


def load_train_partition(name, descriptor):
    rows = read(descriptor['rows']['path'])
    require(len(rows) == descriptor['region_count'], 'Teacher source row count differs')
    validate_rows(rows, list(FAMILIES), descriptor['tile_count'], name)
    tiles = np.memmap(descriptor['tiles']['path'], mode='r', dtype=np.float32, shape=tuple(descriptor['shape']))
    data = {'tiles': tiles, 'rows': rows, 'families': list(FAMILIES)}
    validate_data(data, descriptor, name)
    return data


def validate_data(data, descriptor, name):
    tiles, rows = data['tiles'], data['rows']
    require(isinstance(tiles, np.ndarray) and tiles.dtype == np.float32 and list(tiles.shape) == descriptor['shape']
        and len(rows) == descriptor['region_count'] and rows == read(descriptor['rows']['path']),
        'Extraction TRAIN array, row metadata or shape differs')
    validate_rows(rows, data['families'], len(tiles), name)
    for row in rows:
        block = tiles[row['tile_start']:row['tile_start']+row['tile_count']]
        require(bool(np.isfinite(block).all()) and bool(((block >= 0) & (block <= 1)).all())
            and hashlib.sha256(block.tobytes()).hexdigest() == row.get('tiles_sha256'),
            'TRAIN region tile hash, pixel range or finiteness differs')


def validate_model(model, teacher):
    import torch
    from region_network import RegionFontClassifier
    require(type(model) is RegionFontClassifier and not model.training
        and all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in model.parameters())
        and all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        and state_sha(model.state_dict()) == teacher['state_sha256']
        and parameter_groups(model.state_dict()) == teacher['parameter_groups_sha256'],
        'Teacher must be the complete, frozen eval b08 float32 CNN')


def load_teacher(identity):
    import torch
    from region_network import RegionFontClassifier
    teacher = identity['teacher']; checkpoint = torch.load(teacher['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(checkpoint.get('families') == identity['families'] and checkpoint.get('architecture') == ARCHITECTURE
        and checkpoint.get('selection_sha256') == teacher['selection']['sha256'],
        'Teacher checkpoint schema or bound selection differs')
    model = RegionFontClassifier(25); expected = model.state_dict(); state = checkpoint['state_dict']
    require(set(state) == set(expected) and all(state[k].shape == expected[k].shape and state[k].dtype == torch.float32
        and bool(torch.isfinite(state[k]).all()) for k in expected), 'Incomplete or altered teacher architecture')
    model.load_state_dict(state, strict=True); model.requires_grad_(False); model.eval()
    validate_model(model, teacher)
    return model


def load_cache(folder, identity=None):
    """Return ({original/supplement/known: {base_logits: readonly mmap}}, manifest)."""
    folder = Path(folder).resolve(); manifest = read(folder/'CACHE_MANIFEST.json'); freeze = read(folder/'CACHE_FREEZE.json')
    expected = manifest['identity'] if identity is None else identity
    validate_identity(expected)
    require(manifest.get('schema') == CACHE_SCHEMA and manifest.get('identity') == expected
        and manifest.get('cache_freeze_sha256') == sha(folder/'CACHE_FREEZE.json')
        and freeze.get('schema') == FREEZE_SCHEMA and freeze.get('identity') == expected
        and freeze.get('optimizer_steps_executed') == 0 and freeze.get('batch_size') == BATCH_SIZE
        and manifest.get('optimizer_steps_executed') == 0 and manifest.get('teacher_logits_recomputed') is True
        and manifest.get('teacher_state_before_sha256') == manifest.get('teacher_state_after_sha256') == TEACHER_STATE_SHA
        and manifest.get('teacher_parameter_groups_before_sha256') == manifest.get('teacher_parameter_groups_after_sha256')
            == expected['teacher']['parameter_groups_sha256']
        and manifest.get('tiles_inferred') == expected['total_tiles']
        and set(manifest.get('partitions', {})) == set(PARTITIONS), 'Invalid frozen teacher cache contract')
    arrays = {}
    for name in PARTITIONS:
        count = expected['partitions'][name]['tile_count']; outputs = manifest['partitions'][name]
        require(set(outputs) == {'base_logits'}, 'Teacher cache must contain logits only')
        item = outputs['base_logits']; relative = name+'/base_logits.npy'; path = folder/relative
        require(item.get('path') == relative and path.resolve().parent == folder/name
            and item.get('shape') == [count, 25] and item.get('dtype') == 'float32'
            and sha(path) == item.get('sha256'), 'Teacher logits shape or byte identity changed')
        logits = np.load(path, mmap_mode='r', allow_pickle=False)
        require(logits.dtype == np.float32 and logits.shape == (count, 25) and bool(np.isfinite(logits).all()),
            'Nonfinite or invalid teacher logits')
        arrays[name] = {'base_logits': logits}
    return arrays, manifest


def extract_cache(folder, model, datasets, identity, device):
    import torch
    folder = Path(folder).resolve(); validate_identity(identity)
    require(set(datasets) == set(PARTITIONS) and device == identity['inference_device'], 'Extraction source routing differs')
    for name in PARTITIONS:
        validate_data(datasets[name], identity['partitions'][name], name)
    validate_model(model, identity['teacher'])
    if folder.exists():
        return load_cache(folder, identity)
    before = state_sha(model.state_dict()); groups = parameter_groups(model.state_dict())
    folder.mkdir(parents=True)
    dump(folder/'CACHE_FREEZE.json', {'schema': FREEZE_SCHEMA, 'identity': identity,
        'torch_version': str(torch.__version__), 'batch_size': BATCH_SIZE, 'optimizer_steps_executed': 0})
    freeze_sha = sha(folder/'CACHE_FREEZE.json')
    model.to(device); partitions = {}; inferred = 0
    with torch.inference_mode():
        for name in PARTITIONS:
            count = identity['partitions'][name]['tile_count']; directory = folder/name; directory.mkdir()
            path = directory/'base_logits.npy'
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, 25))
            for start in range(0, count, BATCH_SIZE):
                batch = torch.from_numpy(np.array(datasets[name]['tiles'][start:start+BATCH_SIZE], copy=True)).to(device)
                # Run the actual b08 CNN for this exact source tile interval; no old cache is read.
                logits, _ = model(batch)
                require(logits.dtype == torch.float32 and tuple(logits.shape) == (len(batch), 25)
                    and bool(torch.isfinite(logits).all()), 'Invalid freshly inferred teacher logits')
                array[start:start+len(batch)] = logits.cpu().numpy(); inferred += len(batch)
                if start % (BATCH_SIZE*100) == 0:
                    print(json.dumps({'partition': name, 'tiles_inferred': start+len(batch), 'tiles': count}), flush=True)
            array.flush(); del array
            partitions[name] = {'base_logits': {'path': name+'/base_logits.npy', 'sha256': sha(path),
                'shape': [count, 25], 'dtype': 'float32'}}
    model.cpu(); validate_model(model, identity['teacher'])
    validate_identity(identity)
    for name in PARTITIONS:
        validate_data(datasets[name], identity['partitions'][name], name)
    require(sha(folder/'CACHE_FREEZE.json') == freeze_sha and inferred == identity['total_tiles'],
        'Cache freeze changed or not all TRAIN tiles were inferred')
    manifest = {'schema': CACHE_SCHEMA, 'identity': identity, 'cache_freeze_sha256': freeze_sha,
        'optimizer_steps_executed': 0, 'teacher_logits_recomputed': True, 'tiles_inferred': inferred,
        'teacher_state_before_sha256': before, 'teacher_state_after_sha256': state_sha(model.state_dict()),
        'teacher_parameter_groups_before_sha256': groups,
        'teacher_parameter_groups_after_sha256': parameter_groups(model.state_dict()), 'partitions': partitions}
    dump(folder/'CACHE_MANIFEST.json', manifest)
    return load_cache(folder, identity)


def prepare(args):
    import torch
    require(args.device in ('cpu', 'mps') and (args.device != 'mps' or torch.backends.mps.is_available()),
        'Requested teacher inference device unavailable')
    roots = {'original': args.data, 'supplement': args.supplement, 'known': args.known}
    identity = cache_identity(args.checkpoint, roots, args.device)
    if args.output.exists():
        return load_cache(args.output, identity)
    datasets = {name: load_train_partition(name, identity['partitions'][name]) for name in PARTITIONS}
    model = load_teacher(identity)
    return extract_cache(args.output, model, datasets, identity, args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_ROOTS['original'])
    parser.add_argument('--supplement', type=Path, default=DATA_ROOTS['supplement'])
    parser.add_argument('--known', type=Path, default=DATA_ROOTS['known'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/unified-font-v2/run-source-balanced-v1/model.pth')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/unified-font-v2/teacher-preserve-cache-v1')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    args = parser.parse_args(); _, manifest = prepare(args)
    print(json.dumps({'output': str(args.output.resolve()), 'manifest_sha256': sha(args.output/'CACHE_MANIFEST.json'),
        'tiles_inferred': manifest['tiles_inferred'], 'teacher_state_unchanged':
        manifest['teacher_state_before_sha256'] == manifest['teacher_state_after_sha256']}), flush=True)


if __name__ == '__main__':
    main()
