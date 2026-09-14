"""Restore R21 teacher targets only for original, true-unknown TRAIN rows."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from train_regions import require, sha
from prepare_unified_regions import FAMILIES
from cache_r22_train_teacher import load_cache, BASE_RUN, BASE_SELECTION_SHA
from retention_r21_teacher_cache import (FOLDER as R21_FOLDER, REPORT_SHA, FREEZE_SHA,
    CHECKPOINT_SHA, SELECTION_SHA, STATE_SHA)
from cache_unified_student_teacher import ORDER

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = 24
EXPECTED_ORIGINAL_TILES = 130854
EXPECTED_ORIGINAL_UNKNOWN_TILES = 8072
POLICY = {
    'original_true_unknown': 'pinned original R21 same-tile TRAIN logits',
    'original_known': 'R22', 'supplement_all': 'R22', 'merged_known_all': 'R22',
    'selection_input': 'verified TRAIN source partition and true target only',
    'prediction_routing': False, 'runtime_routing': False,
    'r21_supplement_logits_read': False, 'r21_known_logits_read': False,
}

def read(path):
    return json.loads(Path(path).read_text())

def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def mix_partition(rows, r22_logits, r21_logits, families=FAMILIES):
    """Pure mixing for the original partition; actual truth is verified by its loader."""
    require(list(families) == FAMILIES and isinstance(rows, (list, tuple)) and rows,
            'Invalid class order or missing TRAIN rows')
    require(all(isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 25
                and a.dtype == np.float32 and not a.flags.writeable and np.isfinite(a).all()
                for a in (r22_logits, r21_logits))
            and r22_logits.shape == r21_logits.shape, 'Invalid cached teacher arrays')
    labels = []; cursor = 0; n = len(r22_logits)
    for row in rows:
        require(isinstance(row, dict) and row.get('split') == 'train'
                and row.get('native_font_verified') is True
                and type(row.get('target')) is int and 0 <= row['target'] < 25
                and row.get('family') == families[row['target']]
                and type(row.get('tile_start')) is int and row['tile_start'] == cursor
                and type(row.get('tile_count')) is int and 1 <= row['tile_count'] <= 8
                and cursor + row['tile_count'] <= n, 'TRAIN truth or contiguous tile order changed')
        labels.extend([row['target']] * row['tile_count']); cursor += row['tile_count']
    require(cursor == n, 'TRAIN rows leave tiles unclaimed')
    labels = np.asarray(labels, dtype=np.int64); mask = labels == UNKNOWN
    out = np.where(mask[:, None], r21_logits, r22_logits)
    out.setflags(write=False); labels.setflags(write=False); mask.setflags(write=False)
    return out, labels, mask

def load_replay_teacher(r22_root, datasets):
    """Read only R21 original logits; keep the new unknown and known caches intact."""
    r22_root = Path(r22_root).resolve()
    r22, manifest = load_cache(r22_root, datasets)
    require(set(datasets) == set(r22) == {'original', 'supplement', 'known'}, 'TRAIN source union changed')
    selection_path = BASE_RUN / 'SELECTION.json'
    require(sha(selection_path) == BASE_SELECTION_SHA, 'R22 selection changed')
    selection = read(selection_path)
    bindings = dict(manifest['bindings'])
    def bind(path, digest):
        path = Path(path).resolve(); key = str(path)
        require(sha(path) == digest and (key not in bindings or bindings[key] == digest),
                'Restored teacher proof binding changed')
        bindings[key] = digest
        return path
    for path in (Path(__file__), ROOT/'training/retention_r21_teacher_cache.py',
                 ROOT/'training/cache_unified_student_teacher.py', selection_path,
                 r22_root/'CACHE_MANIFEST.json', r22_root/'CACHE_FREEZE.json'):
        bind(path, sha(path))
    for part in manifest['partitions'].values():
        for item in part['files'].values(): bind(r22_root/item['path'], item['sha256'])
    report_path = bind(R21_FOLDER/'REPORT.json', REPORT_SHA)
    freeze_path = bind(R21_FOLDER/'INFERENCE_FREEZE.json', FREEZE_SHA)
    report, freeze = read(report_path), read(freeze_path)
    require(report['schema'] == 'flux-glyph-train-inference-diagnostic-v1'
            and freeze['schema'] == 'flux-glyph-train-diagnostic-freeze-v1'
            and report['freeze_sha256'] == FREEZE_SHA and freeze['device'] == 'mps'
            and report['optimizer_steps'] == freeze['optimizer_steps'] == 0
            and all(report[k] is False and freeze[k] is False
                    for k in ('calibration_read', 'development_read', 'test_read'))
            and freeze['families'] == selection['families'] == FAMILIES, 'R21 TRAIN-only proof changed')
    identity = freeze['checkpoints']['r21']
    require(identity['checkpoint']['sha256'] == CHECKPOINT_SHA
            and identity['selection']['sha256'] == SELECTION_SHA
            and report['models']['r21']['state_sha256'] == STATE_SHA, 'R21 teacher identity changed')
    for item in [*identity.values(), freeze['source']]: bind(item['path'], item['sha256'])
    r21_selection = read(identity['selection']['path'])
    require(r21_selection['state_after_sha256'] == STATE_SHA
            and r21_selection['families'] == FAMILIES, 'R21 state or class order changed')
    desc = freeze['partitions']['original']; data = datasets['original']
    original_output = report['models']['r21']['partitions']['original']
    require(desc['split'] == 'train' and desc['order'] == ORDER
            and desc['data_manifest']['sha256'] == data['manifest_sha256']
            and desc['partition_manifest']['sha256'] == data['partition_sha256']
            and desc['tile_count'] == original_output['tiles'] == len(data['tiles']) == EXPECTED_ORIGINAL_TILES,
            'Original R21 and R22 TRAIN partition differs')
    for key in ('data_manifest', 'partition_manifest', 'rows', 'tiles'):
        item = desc[key]; path = bind(item['path'], item['sha256'])
        require(selection['bindings'].get(str(path)) == item['sha256'], 'Original proof differs from R22 lineage')
    require(read(desc['rows']['path']) == data['rows'], 'In-memory TRAIN rows differ from the pinned R21 order')
    item = original_output['logits']; out_path = bind(item['path'], item['sha256'])
    require(selection['bindings'].get(str(out_path)) == item['sha256'], 'R21 original logits differ from R22 lineage')
    # Never open the historical 3,220-tile known cache or the R21 supplement cache.
    r21_original = np.load(out_path, mmap_mode='r', allow_pickle=False)
    logits, labels, mask = mix_partition(data['rows'], r22['original']['logits'], r21_original)
    expected = selection['teacher_mix']['original']
    require(_digest(labels) == expected['labels_sha256'] and int(mask.sum()) == expected['unknown_teacher_tiles'] == EXPECTED_ORIGINAL_UNKNOWN_TILES,
            'Restored original unknown target identities changed')
    mixed = {'original': {'logits': logits, 'log_em_ratio': r22['original']['log_em_ratio']},
             'supplement': r22['supplement'], 'known': r22['known']}
    parts = {}
    for name in ('original', 'supplement', 'known'):
        data = datasets[name]
        y = np.concatenate([np.full(r['tile_count'], r['target'], np.int64) for r in data['rows']])
        require(len(y) == len(mixed[name]['logits']) and (name != 'known' or bool((y != UNKNOWN).all())),
                'Retained partition changed or merged-known contains unknown labels')
        parts[name] = {'tiles': len(y), 'unknown_tiles': int((y == UNKNOWN).sum()),
            'restored_r21_tiles': int(mask.sum()) if name == 'original' else 0,
            'labels_sha256': _digest(y), 'r22_logits_sha256': _digest(r22[name]['logits']),
            'mixed_logits_sha256': _digest(mixed[name]['logits']),
            'log_em_ratio_sha256': _digest(mixed[name]['log_em_ratio'])}
    proof = {'schema': 'flux-glyph-original-unknown-teacher-restoration-v1', 'policy': POLICY,
        'families': FAMILIES, 'model_inference': False, 'optimizer_steps': 0,
        'calibration_read': False, 'development_read': False, 'test_read': False,
        'r21_state_sha256': STATE_SHA, 'r21_original_logits_sha256': sha(out_path),
        'partitions': parts, 'bindings': bindings}
    require(all(sha(path) == digest for path, digest in bindings.items()), 'Teacher proof changed while loading')
    return mixed, proof
