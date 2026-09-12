#!/usr/bin/env python3
"""Export a CAL-selected verifier, retaining both published R19 networks verbatim."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import dump, sha, require, state_sha
from train_sans_verifier import POLICY, observations
from calibrate_sans_verifier import TRAINING_SCHEMA, SCHEMA as CALIBRATION_SCHEMA, validate_calibrated_selection
from calibrate_sans_blocking import SCHEMA as BLOCKING_SCHEMA, effective_policy, summarize_selection, validate_blocking_selection


def validate(args):
    selection_path = args.run/'SELECTION.json'
    selection = json.loads(selection_path.read_text())
    require(selection.get('schema') in (TRAINING_SCHEMA, CALIBRATION_SCHEMA, BLOCKING_SCHEMA)
            and selection.get('calibration_passed') is True and selection.get('test_read') is False
            and (selection.get('schema') != TRAINING_SCHEMA or selection.get('policy') == POLICY),
            'selection must pass unchanged CAL policy before export, or its separately declared voting calibration')
    policy = effective_policy(selection)
    bindings = selection.get('bindings')
    require(isinstance(bindings, dict) and bindings and all(isinstance(path, str) and str(Path(path).resolve()) == path
            for path in bindings), 'training bindings must use canonical absolute paths')
    required = [args.data/'MANIFEST.json', args.data/'train/MANIFEST.json', args.data/'calibration/MANIFEST.json',
                args.primary/'metadata.json', args.primary/'model.onnx', args.primary/'rejection.onnx',
                ROOT/'training/train_sans_verifier.py', ROOT/'training/prepare_sans_views.py',
                ROOT/'training/region_network.py', ROOT/'training/network.py']
    require(all(str(path.resolve()) in bindings for path in required), 'export inputs differ from training bindings')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in bindings.items()), 'bound training input changed')
    require(sha(args.run/'CALIBRATION_DECISIONS.json') == selection['calibration_decisions_sha256']
            and sha(args.run/'BASELINE.json') == selection['baseline_sha256'], 'saved calibration changed')
    data = json.loads((args.data/'MANIFEST.json').read_text())
    primary = json.loads((args.primary/'metadata.json').read_text())
    families = selection.get('families')
    require(isinstance(families, list) and len(families) == 9 and len(set(families)) == 9
            and families[-1] == 'Roboto' and data.get('families') == families
            and primary.get('families') == families[:-1], 'training/data/primary family order differs')
    require(primary.get('algorithm') == 'region-cnn64x256-rejection-v2'
            and primary.get('model', {}).get('path') == 'model.onnx'
            and primary.get('model', {}).get('sha256') == sha(args.primary/'model.onnx')
            and primary.get('rejection', {}).get('model', {}).get('path') == 'rejection.onnx'
            and primary.get('rejection', {}).get('model', {}).get('sha256') == sha(args.primary/'rejection.onnx'),
            'bound primary font/rejection model metadata differs')
    decisions = json.loads((args.run/'CALIBRATION_DECISIONS.json').read_text())
    baseline = json.loads((args.run/'BASELINE.json').read_text())
    require(decisions.get('families') == families and decisions.get('policy') == policy
            and baseline.get('families') == primary['families'], 'saved CAL family/policy order differs')
    if selection['schema'] == CALIBRATION_SCHEMA:
        require(validate_calibrated_selection(selection, args.run, args.data, args.primary) == policy,
                'separate voting calibration failed its lineage audit')
    if selection['schema'] == BLOCKING_SCHEMA:
        require(validate_blocking_selection(selection, args.run, args.data, args.primary) == policy,
                'separate product-objective calibration failed its lineage audit')
    return selection


def main(args):
    require(not args.output.exists() and not (args.run/'PARITY.json').exists(), 'export output must be new')
    selection = validate(args)
    policy = effective_policy(selection)
    checkpoint_sha = sha(args.run/'verifier.pth')
    source_bindings = {str(path.resolve()): sha(path) for path in (
        Path(__file__), ROOT/'training/export_region_stable.py', ROOT/'training/train_regions.py',
        ROOT/'training/calibrate_sans_verifier.py', ROOT/'training/calibrate_sans_blocking.py')}
    import torch
    import onnx
    import onnxruntime as ort
    from region_network import RegionFontClassifier
    from export_region_stable import replace_groupnorm
    from prepare_sans_views import load_views
    checkpoint = torch.load(args.run/'verifier.pth', map_location='cpu', weights_only=True)
    require(checkpoint['selection_sha256'] == sha(args.run/'SELECTION.json')
            and checkpoint['families'] == selection['families']
            and state_sha(checkpoint['state_dict']) == selection['state_after_sha256'], 'checkpoint selection binding changed')
    cal = load_views(args.data, 'calibration')
    expected_obs = json.loads((args.run/'CALIBRATION_DECISIONS.json').read_text())['records']
    baseline = json.loads((args.run/'BASELINE.json').read_text())
    primary = json.loads((args.primary/'metadata.json').read_text())
    require(cal['families'] == checkpoint['families'] == selection['families'], 'CAL/checkpoint family order differs')
    require(primary['algorithm'] == 'region-cnn64x256-rejection-v2', 'primary must retain the R19 rejection network')
    reference = RegionFontClassifier(len(checkpoint['families'])).eval()
    reference.load_state_dict(checkpoint['state_dict'], strict=True)
    converted = copy.deepcopy(reference)
    require(replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()), 'export changed frozen parameters')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    output = args.output/'verifier.onnx'
    torch.onnx.export(converted, torch.zeros(2,1,64,256), output, input_names=['tiles'],
                      output_names=['logits','log_em_ratio'], dynamic_axes={'tiles':{0:'batch'},'logits':{0:'batch'},'log_em_ratio':{0:'batch'}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    actual_blocks, expected_blocks, max_logit, max_size = [], [], 0., 0.
    with torch.inference_mode():
        for start in range(0, len(cal['tiles']), 128):
            block = np.array(cal['tiles'][start:start+128], copy=True)
            logits, sizes = reference(torch.from_numpy(block))
            actual, actual_sizes = session.run(None, {'tiles':block})
            np.testing.assert_allclose(actual, logits.numpy(), atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(actual_sizes, sizes.numpy(), atol=2e-4, rtol=2e-4)
            require(np.isfinite(actual).all() and np.isfinite(actual_sizes).all(), 'nonfinite exported outputs')
            max_logit = max(max_logit, float(np.max(np.abs(actual-logits.numpy()))))
            max_size = max(max_size, float(np.max(np.abs(actual_sizes-sizes.numpy()))))
            actual_blocks.append(actual)
            expected_blocks.append(logits.numpy())
    actual_logits, expected_logits = np.concatenate(actual_blocks), np.concatenate(expected_blocks)
    actual_obs = observations(actual_logits, cal['rows'], policy['temperature'], policy['gates'])
    reference_obs = observations(expected_logits, cal['rows'], policy['temperature'], policy['gates'])
    for left, right in ((actual_obs,reference_obs), (actual_obs,expected_obs)):
        require(len(left)==len(right), 'CAL observation count differs')
        require([(r['predicted'],r['passed']) for r in left] == [(r['predicted'],r['passed']) for r in right],
                'export changed a frozen CAL decision')
        np.testing.assert_allclose([r['probabilities'] for r in left], [r['probabilities'] for r in right], atol=2e-5, rtol=1e-4)
    metrics, _ = summarize_selection(selection, cal['rows'], baseline['records'], actual_obs, checkpoint['families'], baseline['families'])
    require(metrics == selection['selected']['metrics'], 'export changed selection metrics')
    indices = np.unique(np.linspace(0,len(cal['tiles'])-1,512).astype(int))
    sample = np.array(cal['tiles'][indices],copy=True)
    batches = []
    for batch_size in (1,7,32,128):
        actual = np.concatenate([session.run(['logits'],{'tiles':sample[start:start+batch_size]})[0]
                                 for start in range(0,len(sample),batch_size)])
        np.testing.assert_allclose(actual, expected_logits[indices], atol=2e-4, rtol=2e-4)
        batches.append({'batch_size':batch_size,'samples':len(sample),'passed':True})
    require(validate(args) == selection and sha(args.run/'verifier.pth') == checkpoint_sha,
            'selection/checkpoint changed during export')
    require(all(sha(path) == digest for path, digest in source_bindings.items()), 'export/lowering source changed during export')
    for name in ('model.onnx', 'rejection.onnx'):
        shutil.copyfile(args.primary/name, args.output/name)
        require(sha(args.output/name)==sha(args.primary/name), 'existing model changed')
    primary['algorithm'] = 'region-cnn64x256-consensus-v3'
    primary['verifier'] = {
        'schema':'flux-glyph-region-verifier-v1','algorithm':'region-font-verifier-cnn64x256-v1',
        'model':{'path':'verifier.onnx','sha256':sha(output)}, 'families':checkpoint['families'],
        'temperature':policy['temperature'],'gates':policy['gates'], 'base_model_sha256':primary['model']['sha256'],
        'training':{'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,
                    'selection_schema':selection['schema'], 'optimizer_steps_in_selection_stage':selection.get('optimizer_steps_executed'),
                    'weights_unchanged_in_selection_stage':selection.get('weights_unchanged',False),
                    'native_data_and_views':selection['bindings'], 'test_used_for_training':False}}
    dump(args.output/'metadata.json',primary)
    from flux_glyph.region_font import RegionFontClassifier as Runtime
    Runtime(args.output)
    parity = {'schema':'flux-glyph-sans-verifier-parity-v1','passed':True,'test_read':False,
              'model_sha256':sha(output),'metadata_sha256':sha(args.output/'metadata.json'),
              'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,
              'calibration_regions':len(cal['rows']),'calibration_tiles':len(cal['tiles']),
              'max_logit_absolute_error':max_logit,'max_unused_size_absolute_error':max_size,
              'calibration_decisions_identical':True,'original_torch_groupnorm_reference':True,'batches':batches,
              'export_source_sha256':source_bindings[str(Path(__file__).resolve())], 'source_bindings':source_bindings,
              'primary_model_sha256':primary['model']['sha256'],
              'rejection_model_sha256':primary['rejection']['model']['sha256'],
              'primary_and_rejection_unchanged':True}
    require(validate(args) == selection and sha(args.run/'verifier.pth') == checkpoint_sha
            and all(sha(path) == digest for path, digest in source_bindings.items()),
            'selection/checkpoint/export source changed before parity was frozen')
    dump(args.run/'PARITY.json',parity)
    print(json.dumps(parity,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','primary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
