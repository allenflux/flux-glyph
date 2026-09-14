#!/usr/bin/env python3
"""Freeze R22 predictions on the exact existing TRAIN tiles, without gradients."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import require, sha, state_sha
from prepare_unified_regions import FAMILIES, load_split
from prepare_unified_unknown_supplement import load_supplement
from merge_unified_weight_pairs import load_merged

BASE_RUN = ROOT / 'artifacts/unified-font-v3/run-wide-micro-recovery-v1'
BASE_SHA = '53d73321c109825014fca96435f08a8c7a01ec6fe2abefa727855e1500d94c42'
BASE_SELECTION_SHA = '49bd22a905bda9ffda11822323c81423482085e10adb477aed90089fdf43618e'
BASE_STATE_SHA = 'ea89337ebd8df8b71833073effbe78180e80b6fe41063b993df27c24a62ad97a'
ARCHITECTURE = 'region-cnn64x256-unified-wide-v1'
SOURCES = {
    'original': (ROOT / 'artifacts/unified-font-v1/data-v1',
                 'dfdd9fa746b28b3665eed10491ce83c5c872f16b2499fe3ae6c6e6bb9ed40a23', 130854),
    'supplement': (ROOT / 'artifacts/unified-font-v2/new-unknown-capture-v1/data',
                   '43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05', 3156),
    'known': (ROOT / 'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/merged-data',
              'ec9b616538444741742f91414c7e66ae51bdf036c3956fdca9d6bdbbb7873b83', 4820),
}
TOTAL_TILES = sum(source[2] for source in SOURCES.values())


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_base():
    import torch
    from wide_region_network import WideRegionFontClassifier
    require(sha(BASE_RUN / 'model.pth') == BASE_SHA
            and sha(BASE_RUN / 'SELECTION.json') == BASE_SELECTION_SHA, 'R22 initializer changed')
    selection = read(BASE_RUN / 'SELECTION.json')
    require(selection['promotion_allowed'] is True and selection['families'] == FAMILIES
            and selection['state_after_sha256'] == BASE_STATE_SHA
            and len(selection['selected']['retention_checks']) == 46
            and all(row['passed'] for row in selection['selected']['retention_checks'])
            and selection['test_read'] is False, 'R22 baseline does not have the recorded acceptance')
    checkpoint = torch.load(BASE_RUN / 'model.pth', map_location='cpu', weights_only=True)
    require(checkpoint['selection_sha256'] == BASE_SELECTION_SHA
            and checkpoint['architecture'] == ARCHITECTURE and checkpoint['families'] == FAMILIES
            and state_sha(checkpoint['state_dict']) == BASE_STATE_SHA, 'R22 checkpoint state differs')
    model = WideRegionFontClassifier(25).cpu().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, selection


def load_training():
    """Existing strict loaders verify source proofs and partition byte hashes."""
    result = {}
    for name, (root, digest, count) in SOURCES.items():
        require(sha(root / 'MANIFEST.json') == digest, 'Changed TRAIN root: ' + name)
        data = (load_split(root, 'train') if name == 'original' else
                load_supplement(root) if name == 'supplement' else load_merged(root))
        require(data['families'] == FAMILIES and data['partition']['split'] == 'train'
                and len(data['tiles']) == count, 'Changed TRAIN geometry or class order: ' + name)
        cursor = 0
        for row in data['rows']:
            require(row['split'] == 'train' and row['native_font_verified'] is True
                    and row['tile_start'] == cursor and FAMILIES[row['target']] == row['family'],
                    'Non-TRAIN or relabeled teacher tile')
            cursor += row['tile_count']
        require(cursor == count, 'Orphan TRAIN teacher tiles')
        result[name] = data
    validate_datasets(result)
    return result


def validate_datasets(datasets):
    require(set(datasets) == set(SOURCES), 'Teacher datasets must be the three exact TRAIN roots')
    for name, data in datasets.items():
        root, digest, count = SOURCES[name]
        part = read(root / 'train/MANIFEST.json')
        require(sha(root / 'MANIFEST.json') == digest and data['partition'] == part
                and data['families'] == FAMILIES and part['split'] == 'train'
                and data['rows'] == read(root / 'train' / part['metadata']['path']),
                'In-memory TRAIN row order or labels differ from the bound partition')
        array = data['tiles']
        require(isinstance(array, np.memmap) and array.mode == 'r'
                and Path(array.filename).resolve() == (root / 'train' / part['array']['path']).resolve()
                and array.shape == (count, 1, 64, 256) and array.dtype == np.float32,
                'TRAIN tiles must use the exact read-only partition mapping')


def source_bindings(datasets):
    paths = [Path(__file__), ROOT / 'training/wide_region_network.py',
             ROOT / 'training/train_regions.py', ROOT / 'training/prepare_unified_regions.py',
             ROOT / 'training/prepare_unified_unknown_supplement.py',
             ROOT / 'training/merge_unified_weight_pairs.py',
             BASE_RUN / 'model.pth', BASE_RUN / 'SELECTION.json', BASE_RUN / 'TRAINING_FREEZE.json']
    for name, data in datasets.items():
        root = SOURCES[name][0]
        paths.extend([root / 'MANIFEST.json', root / 'train/MANIFEST.json'])
        paths.extend(root / 'train' / data['partition'][key]['path'] for key in ('metadata', 'array'))
    return {str(path.resolve()): sha(path) for path in paths}


def cache(output, device):
    import torch
    output = Path(output).resolve()
    require(not output.exists(), 'Preserve prior teacher caches; use a new output directory')
    torch.set_num_threads(4)
    if device == 'mps':
        require(torch.backends.mps.is_available(), 'Local MPS unavailable')
    datasets = load_training()
    model, _ = load_base()
    bindings = source_bindings(datasets)
    output.mkdir(parents=True)
    freeze = {'schema': 'flux-glyph-r22-train-teacher-freeze-v1', 'bindings': bindings,
              'checkpoint_sha256': BASE_SHA, 'selection_sha256': BASE_SELECTION_SHA,
              'state_sha256': BASE_STATE_SHA, 'architecture': ARCHITECTURE,
              'families': FAMILIES, 'device': device,
              'libraries': {'torch': str(torch.__version__), 'numpy': np.__version__,
                            'python': sys.version.split()[0]},
              'batch_size': 128, 'optimizer_steps': 0, 'calibration_read': False,
              'development_read': False, 'test_read': False, 'model_count': 1,
              'scope': 'Only original, unknown-supplement and merged-known TRAIN image tiles.'}
    write_new(output / 'CACHE_FREEZE.json', freeze)
    freeze_sha = sha(output / 'CACHE_FREEZE.json')
    model.to(device).eval()
    partitions = {}
    started = time.monotonic()
    with torch.inference_mode():
        for name, data in datasets.items():
            folder = output / name
            folder.mkdir()
            n = len(data['tiles'])
            logits = np.lib.format.open_memmap(folder / 'logits.npy', mode='w+', dtype=np.float32, shape=(n, 25))
            ratios = np.lib.format.open_memmap(folder / 'log_em_ratio.npy', mode='w+', dtype=np.float32, shape=(n,))
            for start in range(0, n, 128):
                block = torch.from_numpy(np.array(data['tiles'][start:start + 128], copy=True)).to(device)
                font, size = model(block)
                a, b = font.cpu().numpy(), size.cpu().numpy()
                require(np.isfinite(a).all() and np.isfinite(b).all() and (np.abs(b) <= 3).all(),
                        'Nonfinite teacher outputs')
                logits[start:start + len(a)] = a
                ratios[start:start + len(b)] = b
                if start % 8192 == 0:
                    print(json.dumps({'source': name, 'tiles': min(start + 128, n), 'total': n,
                                      'seconds': round(time.monotonic() - started, 1)}), flush=True)
            logits.flush(); ratios.flush()
            del logits, ratios
            partitions[name] = {'tiles': n, 'partition_sha256': sha(SOURCES[name][0] / 'train/MANIFEST.json'),
                                'files': {key: {'path': f'{name}/{key}.npy', 'sha256': sha(folder / f'{key}.npy')}
                                          for key in ('logits', 'log_em_ratio')}}
    model.cpu()
    require(state_sha(model.state_dict()) == BASE_STATE_SHA
            and sha(output / 'CACHE_FREEZE.json') == freeze_sha
            and all(sha(path) == digest for path, digest in bindings.items()), 'Teacher or source mutated during inference')
    report = {**freeze, 'schema': 'flux-glyph-r22-train-teacher-cache-v1', 'freeze_sha256': freeze_sha,
              'partitions': partitions, 'total_tiles': sum(v['tiles'] for v in partitions.values()),
              'weights_unchanged': True, 'seconds': round(time.monotonic() - started, 1)}
    write_new(output / 'CACHE_MANIFEST.json', report)
    print(json.dumps({'status': 'COMPLETE', 'tiles': report['total_tiles'],
                      'manifest_sha256': sha(output / 'CACHE_MANIFEST.json')}), flush=True)


def load_cache(output, datasets):
    output = Path(output).resolve()
    validate_datasets(datasets)
    manifest = read(output / 'CACHE_MANIFEST.json')
    freeze = read(output / 'CACHE_FREEZE.json')
    require(manifest['schema'] == 'flux-glyph-r22-train-teacher-cache-v1'
            and manifest['checkpoint_sha256'] == BASE_SHA and manifest['families'] == FAMILIES
            and manifest['selection_sha256'] == BASE_SELECTION_SHA
            and manifest['state_sha256'] == BASE_STATE_SHA and manifest['architecture'] == ARCHITECTURE
            and freeze['schema'] == 'flux-glyph-r22-train-teacher-freeze-v1'
            and manifest['weights_unchanged'] is True and set(manifest['partitions']) == set(SOURCES)
            and set(manifest['libraries']) == {'torch', 'numpy', 'python'}
            and all(isinstance(v, str) and v for v in manifest['libraries'].values())
            and manifest['freeze_sha256'] == sha(output / 'CACHE_FREEZE.json')
            and manifest['total_tiles'] == TOTAL_TILES and manifest['optimizer_steps'] == 0
            and sum(part['tiles'] for part in manifest['partitions'].values()) == manifest['total_tiles']
            and all(manifest.get(key) is False for key in ('calibration_read', 'development_read', 'test_read'))
            and all(manifest[key] == value for key, value in freeze.items() if key != 'schema')
            and manifest['bindings'] == source_bindings(datasets), 'Unbound R22 TRAIN teacher cache')
    result = {}
    for name, data in datasets.items():
        entry = manifest['partitions'][name]
        require(entry['tiles'] == len(data['tiles'])
                and entry['partition_sha256'] == sha(SOURCES[name][0] / 'train/MANIFEST.json'), 'Teacher partition changed')
        arrays = {}
        require(set(entry['files']) == {'logits', 'log_em_ratio'}, 'Unexpected teacher cache files')
        for key, identity in entry['files'].items():
            require(identity['path'] == f'{name}/{key}.npy', 'Unexpected teacher cache path')
            path = output / identity['path']
            require(sha(path) == identity['sha256'], 'Teacher cache bytes differ')
            arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        require(set(arrays) == {'logits', 'log_em_ratio'}
                and arrays['logits'].shape == (len(data['tiles']), 25)
                and arrays['log_em_ratio'].shape == (len(data['tiles']),)
                and all(x.dtype == np.float32 and np.isfinite(x).all() for x in arrays.values()),
                'Teacher output contract differs')
        result[name] = arrays
    return result, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    cache(args.output, args.device)
