#!/usr/bin/env python3
"""Cache only pinned b08 logits for the merged paired-weight TRAIN tensor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
import cache_unified_student_teacher as strict
from merge_unified_weight_pairs import load_merged, SCHEMA as DATA_SCHEMA
from prepare_unified_regions import FAMILIES
from train_regions import dump, require, sha, state_sha

ARCHITECTURE = strict.ARCHITECTURE
CACHE_SCHEMA = 'flux-glyph-unified-weight-teacher-cache-v1'
FREEZE_SCHEMA = 'flux-glyph-unified-weight-teacher-cache-freeze-v1'
BATCH_SIZE = 128
ORDER = 'Exact merged TRAIN tile_start order; no resampling, shuffling or aggregation.'
REQUIRED_SOURCES = ('training/cache_unified_weight_teacher.py', 'training/cache_unified_student_teacher.py',
    'training/merge_unified_weight_pairs.py', 'training/region_network.py', 'training/network.py',
    'training/train_regions.py', 'training/train_unified_retention.py', 'training/prepare_unified_regions.py')


def read(path):
    return json.loads(Path(path).read_text())


def entry(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def _bind(bindings, item):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'Incomplete weight-cache source identity')
    path = Path(item['path']).resolve()
    require(path.is_file() and str(path) == item['path'] and sha(path) == item['sha256']
            and (str(path) not in bindings or bindings[str(path)] == item['sha256']),
            'Weight-cache source binding changed')
    bindings[str(path)] = item['sha256']


def validate_data(data):
    """Validate the already-loaded merged TRAIN bytes and exact contiguous row order."""
    require(isinstance(data, dict) and data.get('families') == list(FAMILIES)
            and data.get('manifest', {}).get('schema') == DATA_SCHEMA
            and data.get('partition', {}).get('schema') == DATA_SCHEMA
            and data['partition'].get('split') == data['manifest'].get('only_split') == 'train',
            'Weight teacher requires the verified merged 25-class TRAIN root')
    tiles, rows, part = data.get('tiles'), data.get('rows'), data['partition']
    require(isinstance(tiles, np.ndarray) and tiles.dtype == np.float32
            and tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256)
            and list(tiles.shape) == part.get('shape') and len(tiles) == part.get('tiles')
            and isinstance(rows, list) and rows and len(rows) == part.get('views'),
            'Merged TRAIN tile shape or row count differs')
    offset = 0; seen = set()
    for row in rows:
        count = row.get('tile_count'); key = (row.get('source_id'), row.get('region_id'), row.get('view'))
        require(row.get('split') == row.get('original_split') == 'train'
                and row.get('native_font_verified') is True and row.get('domain') == 'android'
                and row.get('source_dataset') == 'android_paired_known_supplement'
                and row.get('family') in ('LXGW WenKai', 'WenQuanYi Micro Hei')
                and row.get('source_font_family') == row['family']
                and row.get('target') == list(FAMILIES).index(row['family'])
                and all(isinstance(value, str) and value for value in key) and key not in seen
                and row.get('tile_start') == offset and type(count) is int and 1 <= count <= 8
                and offset + count <= len(tiles), 'Merged TRAIN row was relabeled, reordered or remapped')
        block = tiles[offset:offset+count]
        require(bool(np.isfinite(block).all()) and bool(((block >= 0) & (block <= 1)).all())
                and hashlib.sha256(block.tobytes()).hexdigest() == row.get('tiles_sha256'),
                'Merged TRAIN tile bytes differ from row identity')
        seen.add(key); offset += count
    require(offset == len(tiles), 'Merged TRAIN rows leave tile bytes unclaimed')


def dataset_identity(root, data):
    root = Path(root).resolve(); validate_data(data)
    items = {name: entry(root/path) for name, path in (
        ('manifest', 'MANIFEST.json'), ('preparation_freeze', 'PREPARATION_FREEZE.json'),
        ('partition', 'train/MANIFEST.json'), ('rows', 'train/rows.json'),
        ('tiles', 'train/tiles.raw'), ('source_mapping', 'train/SOURCE_MAPPING.json'))}
    require(items['manifest']['sha256'] == data['manifest_sha256']
            and items['partition']['sha256'] == data['partition_sha256']
            and data['partition']['root_manifest_sha256'] == items['manifest']['sha256']
            and data['partition']['metadata']['sha256'] == items['rows']['sha256']
            and data['partition']['array']['sha256'] == items['tiles']['sha256']
            and data['partition']['source_mapping']['sha256'] == items['source_mapping']['sha256']
            and read(items['rows']['path']) == data['rows'], 'Loaded merged data differs from bound files')
    return {**items, 'region_count': len(data['rows']), 'tile_count': len(data['tiles']),
        'shape': list(data['tiles'].shape), 'order': ORDER}


def _source_bindings(dataset, teacher):
    bindings = {}
    for value in dataset.values():
        if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
            _bind(bindings, value)
    for item in teacher.values():
        if isinstance(item, dict) and set(item) == {'path', 'sha256'}:
            _bind(bindings, item)
    manifest = read(dataset['manifest']['path'])
    for path, digest in manifest.get('bindings', {}).items():
        _bind(bindings, {'path': path, 'sha256': digest})
    for source in manifest.get('sources', {}).values():
        for item in source.values(): _bind(bindings, item)
    for relative in REQUIRED_SOURCES:
        _bind(bindings, entry(ROOT/relative))
    return bindings


def cache_identity(data_root, data, teacher, device):
    require(device in ('cpu', 'mps') and teacher.get('state_sha256') == strict.TEACHER_STATE_SHA
            and teacher.get('checkpoint', {}).get('sha256') == strict.TEACHER_CHECKPOINT_SHA
            and teacher.get('selection', {}).get('sha256') == strict.TEACHER_SELECTION_SHA,
            'Weight cache requires the exact frozen b08 teacher')
    dataset = dataset_identity(data_root, data)
    return {'architecture': ARCHITECTURE, 'families': list(FAMILIES), 'teacher': teacher,
        'dataset': dataset, 'bindings': _source_bindings(dataset, teacher),
        'inference_device': device, 'batch_size': BATCH_SIZE, 'outputs': ['base_logits'],
        'optimizer_steps_executed': 0, 'teacher_optimizer_steps_executed': 0,
        'teacher_model_count': 1, 'teacher_logits_recomputed': True, 'historical_logits_reused': False,
        'tile_count': len(data['tiles']), 'test_read': False, 'development_holdout_read': False,
        'calibration_images_read': False, 'order': ORDER}


def validate_identity(identity, data):
    root = Path(identity['dataset']['manifest']['path']).parent
    expected = cache_identity(root, data, identity['teacher'], identity['inference_device'])
    require(identity == expected, 'Weight-cache teacher, merged data or source closure changed')
    return expected


def load_cache(root, data, teacher, bindings):
    """Return ({base_logits: readonly mmap}, manifest, CACHE_MANIFEST path/SHA)."""
    root = Path(root).resolve(); manifest_path = root/'CACHE_MANIFEST.json'; freeze_path = root/'CACHE_FREEZE.json'
    manifest, freeze = read(manifest_path), read(freeze_path)
    identity = manifest.get('identity', {}); validate_identity(identity, data)
    require(identity.get('teacher') == teacher and manifest.get('schema') == CACHE_SCHEMA
            and freeze.get('schema') == FREEZE_SCHEMA and freeze.get('identity') == identity
            and manifest.get('cache_freeze_sha256') == sha(freeze_path)
            and manifest.get('optimizer_steps_executed') == freeze.get('optimizer_steps_executed') == 0
            and manifest.get('teacher_state_before_sha256') == manifest.get('teacher_state_after_sha256')
                == teacher['state_sha256']
            and manifest.get('teacher_parameter_groups_before_sha256')
                == manifest.get('teacher_parameter_groups_after_sha256') == teacher['parameter_groups_sha256']
            and manifest.get('tiles_inferred') == identity['tile_count'], 'Frozen weight-teacher cache contract differs')
    item = manifest.get('base_logits', {}); path = root/'base_logits.npy'
    require(item == {'path': 'base_logits.npy', 'sha256': sha(path),
            'shape': [identity['tile_count'], 25], 'dtype': 'float32'},
            'Weight-teacher logits path, SHA, shape or dtype differs')
    logits = np.load(path, mmap_mode='r', allow_pickle=False)
    require(isinstance(logits, np.memmap) and not logits.flags.writeable
            and logits.dtype == np.float32 and logits.shape == (identity['tile_count'], 25)
            and bool(np.isfinite(logits).all()), 'Invalid cached weight-teacher logits')
    additions = dict(identity['bindings'])
    for item in (entry(manifest_path), entry(freeze_path), entry(path)): additions[item['path']] = item['sha256']
    require(all(name not in bindings or bindings[name] == digest for name, digest in additions.items()),
            'Caller binding conflicts with weight-teacher cache')
    bindings.update(additions)
    return {'base_logits': logits}, manifest, entry(manifest_path)


def extract_cache(root, model, data, identity, device):
    import torch
    root = Path(root).resolve()
    require(not root.exists() and device == identity['inference_device'],
            'Weight-teacher output must be a fresh path on the declared device')
    validate_identity(identity, data); strict.validate_model(model, identity['teacher'])
    before = state_sha(model.state_dict()); groups = strict.parameter_groups(model.state_dict())
    root.mkdir(parents=True)
    dump(root/'CACHE_FREEZE.json', {'schema': FREEZE_SCHEMA, 'identity': identity,
        'torch_version': str(torch.__version__), 'inference_device': device,
        'batch_size': BATCH_SIZE, 'optimizer_steps_executed': 0,
        'test_read': False, 'development_holdout_read': False, 'calibration_images_read': False})
    path = root/'base_logits.npy'; count = identity['tile_count']
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, 25))
    model.to(device); inferred = 0
    with torch.inference_mode():
        for start in range(0, count, BATCH_SIZE):
            batch = torch.from_numpy(np.array(data['tiles'][start:start+BATCH_SIZE], copy=True)).to(device)
            logits, _ = model(batch)
            require(logits.dtype == torch.float32 and tuple(logits.shape) == (len(batch), 25)
                    and bool(torch.isfinite(logits).all()), 'Invalid freshly inferred b08 logits')
            output[start:start+len(batch)] = logits.cpu().numpy(); inferred += len(batch)
    output.flush(); del output
    model.cpu(); strict.validate_model(model, identity['teacher']); validate_identity(identity, data)
    require(before == state_sha(model.state_dict()) and groups == strict.parameter_groups(model.state_dict())
            and inferred == count, 'Teacher state changed or merged TRAIN tiles were skipped')
    freeze_sha = sha(root/'CACHE_FREEZE.json')
    manifest = {'schema': CACHE_SCHEMA, 'identity': identity, 'cache_freeze_sha256': freeze_sha,
        'base_logits': {'path': 'base_logits.npy', 'sha256': sha(path), 'shape': [count, 25], 'dtype': 'float32'},
        'tiles_inferred': inferred, 'optimizer_steps_executed': 0, 'teacher_logits_recomputed': True,
        'teacher_state_before_sha256': before, 'teacher_state_after_sha256': state_sha(model.state_dict()),
        'teacher_parameter_groups_before_sha256': groups,
        'teacher_parameter_groups_after_sha256': strict.parameter_groups(model.state_dict()),
        'test_read': False, 'development_holdout_read': False, 'calibration_images_read': False}
    dump(root/'CACHE_MANIFEST.json', manifest)
    bindings = {}
    return load_cache(root, data, identity['teacher'], bindings)


def prepare(args):
    import torch
    require(args.device in ('cpu', 'mps') and (args.device != 'mps' or torch.backends.mps.is_available()),
            'Requested weight-teacher inference device unavailable')
    require(not args.output.exists(), 'Refuse to overwrite or reuse a weight-teacher cache')
    data = load_merged(args.data)
    teacher, _ = strict.teacher_identity(args.checkpoint)
    identity = cache_identity(args.data, data, teacher, args.device)
    model = strict.load_teacher(identity)
    return extract_cache(args.output, model, data, identity, args.device)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    task = ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1'
    result.add_argument('--data', type=Path, default=task/'merged-data')
    result.add_argument('--checkpoint', type=Path,
        default=ROOT/'artifacts/unified-font-v2/run-source-balanced-v1/model.pth')
    result.add_argument('--output', type=Path, default=task/'teacher-cache')
    result.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args(); _, manifest, cache = prepare(arguments)
    print(json.dumps({'cache': cache, 'tiles_inferred': manifest['tiles_inferred'],
        'teacher_state_unchanged': manifest['teacher_state_before_sha256'] == manifest['teacher_state_after_sha256']}))
