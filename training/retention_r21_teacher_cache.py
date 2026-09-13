"""Pinned R21 TRAIN logits for offline unknown-label supervision only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from train_regions import sha, require
from cache_unified_student_teacher import DATA_PINS, PARTITIONS, ORDER

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'artifacts/unified-font-v3/train-inference-audit-v1'
REPORT_SHA = '4dc5fe667621f04567a6b89ce7e0f2695f04678defe742148db060fa07b5aa84'
FREEZE_SHA = 'd2b571ef033ac9628754c44647b4dadc13b64c24592cb0e003501bb17ba5eb26'
CHECKPOINT_SHA = '960b210091f7c8b13a7283e6bd7f4a7e48e1bd7a1bf3a8d091d0fe11b4bf367e'
SELECTION_SHA = '5b99f17bfda9566a334abd81cffab9cf19e3da74aa38a20fc2405638248268f9'
STATE_SHA = '247fb483cea5bdf3d7b45b90b0d95f58738fb937a065d19b8a8a9efb1f94a16e'
TEACHER_POLICY = {
    'named_targets': 'b08 frozen source-balanced CNN',
    'unknown_target': 'R21 frozen unified CNN',
    'selection_input': 'true TRAIN target only',
    'mask': 'selected same-tile teacher argmax equals true TRAIN target',
    'temperature': 1., 'kl_weight': 2.,
    'offline_teacher_count': 2, 'teacher_models_resident_during_optimizer': 0,
    'teacher_models_deployed': 0, 'inference_routing': False,
}


def read(path):
    return json.loads(Path(path).read_text())


def load_r21_train_logits(datasets, bindings):
    require(set(datasets) == set(PARTITIONS), 'R21 teacher needs all three verified TRAIN sources')
    report_path, freeze_path = FOLDER/'REPORT.json', FOLDER/'INFERENCE_FREEZE.json'
    require(sha(report_path) == REPORT_SHA and sha(freeze_path) == FREEZE_SHA,
            'Actual R21 TRAIN inference proof changed')
    report, freeze = read(report_path), read(freeze_path)
    require(report['schema'] == 'flux-glyph-train-inference-diagnostic-v1'
            and freeze['schema'] == 'flux-glyph-train-diagnostic-freeze-v1'
            and report['freeze_sha256'] == FREEZE_SHA and freeze['device'] == 'mps'
            and all(report[key] is False and freeze[key] is False
                    for key in ('calibration_read', 'development_read', 'test_read'))
            and report['optimizer_steps'] == freeze['optimizer_steps'] == 0,
            'R21 cached outputs must come from the completed TRAIN-only no-update inference')
    identity = freeze['checkpoints']['r21']
    require(identity['checkpoint']['sha256'] == CHECKPOINT_SHA
            and identity['selection']['sha256'] == SELECTION_SHA
            and report['models']['r21']['state_sha256'] == STATE_SHA,
            'R21 teacher must remain distinct from b08 and the failed confidence-floor candidate')
    selection = read(identity['selection']['path'])
    require(selection['state_after_sha256'] == STATE_SHA
            and selection['families'] == freeze['families'], 'R21 class or full-state identity changed')

    def bind(item):
        path = Path(item['path']).resolve()
        require(sha(path) == item['sha256'] and
                (str(path) not in bindings or bindings[str(path)] == item['sha256']),
                'R21 teacher cache or source binding changed')
        bindings[str(path)] = item['sha256']

    for item in [*identity.values(), freeze['source'],
                 {'path': str(report_path), 'sha256': REPORT_SHA},
                 {'path': str(freeze_path), 'sha256': FREEZE_SHA},
                 {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}]:
        bind(item)
    network_sources={}
    for name in ('training/network.py','training/region_network.py','training/train_regions.py'):
        path=(ROOT/name).resolve()
        require(selection['bindings'].get(str(path))==sha(path),
                'R21 inference implementation differs from the original checkpoint source')
        item={'path':str(path),'sha256':sha(path)}
        bind(item);network_sources[name]=item
    arrays, outputs = {}, report['models']['r21']['partitions']
    require(set(outputs) == set(PARTITIONS), 'R21 cache partition union changed')
    for name in PARTITIONS:
        data, descriptor, output = datasets[name], freeze['partitions'][name], outputs[name]
        require(descriptor['split'] == 'train' and descriptor['order'] == ORDER
                and descriptor['data_manifest']['sha256'] == data['manifest_sha256'] == DATA_PINS[name][0]
                and descriptor['partition_manifest']['sha256'] == data['partition_sha256'] == DATA_PINS[name][1]
                and descriptor['tile_count'] == output['tiles'] == len(data['tiles']) == DATA_PINS[name][3]
                and read(descriptor['rows']['path']) == data['rows']
                and data['families'] == freeze['families'], 'R21 and b08 must address the same verified TRAIN tiles')
        for key in ('data_manifest', 'partition_manifest', 'rows', 'tiles'):
            bind(descriptor[key])
        bind(output['logits'])
        values = np.load(output['logits']['path'], mmap_mode='r', allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (len(data['tiles']), 25)
                and np.isfinite(values).all(), 'Invalid R21 TRAIN logits')
        arrays[name] = values
    return arrays, {**identity, 'state_sha256': STATE_SHA,
                    'inference_report': {'path': str(report_path), 'sha256': REPORT_SHA},
                    'inference_freeze': {'path': str(freeze_path), 'sha256': FREEZE_SHA},
                    'outputs': outputs, 'network_sources':network_sources, 'selection_policy': TEACHER_POLICY}
