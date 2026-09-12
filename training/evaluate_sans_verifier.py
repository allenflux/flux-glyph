#!/usr/bin/env python3
"""Evaluate one frozen v3 verifier on fixed native TEST views; never select weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import dump,require,sha
from train_sans_verifier import observations,baseline_observations
from export_sans_verifier import validate as validate_training_selection
from calibrate_sans_blocking import effective_policy,summarize_selection


def validate(args):
    """Verify all model/data bindings before a held-out array can be opened."""
    selection = validate_training_selection(args)
    parity = json.loads((args.run/'PARITY.json').read_text())
    require(parity.get('schema') == 'flux-glyph-sans-verifier-parity-v1'
            and parity.get('passed') is True and parity.get('test_read') is False
            and parity.get('selection_sha256') == sha(args.run/'SELECTION.json')
            and parity.get('checkpoint_sha256') == sha(args.run/'verifier.pth'),
            'selection/checkpoint/export must be frozen before test')
    bindings = parity.get('source_bindings')
    expected_sources = {str(path.resolve()) for path in (
        ROOT/'training/export_sans_verifier.py', ROOT/'training/export_region_stable.py', ROOT/'training/train_regions.py',
        ROOT/'training/calibrate_sans_verifier.py', ROOT/'training/calibrate_sans_blocking.py')}
    require(isinstance(bindings, dict) and set(bindings) == expected_sources
            and all(sha(path) == digest for path, digest in bindings.items()), 'frozen export/lowering source changed')
    require(parity.get('export_source_sha256') == bindings[str((ROOT/'training/export_sans_verifier.py').resolve())],
            'export source SHA disagrees with frozen provenance')
    metadata = json.loads((args.region/'metadata.json').read_text())
    primary = json.loads((args.primary/'metadata.json').read_text())
    require(sha(args.region/'metadata.json') == parity.get('metadata_sha256')
            and metadata.get('algorithm') == 'region-cnn64x256-consensus-v3'
            and metadata.get('families') == primary['families']
            and metadata.get('verifier', {}).get('families') == selection['families'], 'test model/family order differs from frozen export')
    policy = effective_policy(selection)
    require(metadata['verifier'].get('temperature') == policy['temperature']
            and metadata['verifier'].get('gates') == policy['gates'], 'test verifier thresholds differ from the selected policy')
    require(metadata.get('verifier', {}).get('model', {}).get('path') == 'verifier.onnx'
            and metadata['verifier']['model']['sha256'] == parity.get('model_sha256')
            and sha(args.region/'verifier.onnx') == parity['model_sha256'], 'test verifier differs from frozen export')
    for name, recorded in (('model.onnx', 'primary_model_sha256'), ('rejection.onnx', 'rejection_model_sha256')):
        require(sha(args.region/name) == sha(args.primary/name) == parity.get(recorded), 'test primary/rejection differs from frozen source')
    restored = dict(metadata)
    restored.pop('verifier', None)
    restored['algorithm'] = primary['algorithm']
    require(restored == primary, 'consensus release changed the primary font/rejection contract')
    return selection, parity


def diagnostics(details):
    """Separate genuine additional rejection from the verifier's own recall."""
    roboto = [row for row in details if row['family'] == 'Roboto']
    previous_wrong = [row for row in roboto if row['before_wrong']]
    removed = sum(not row['after_wrong'] for row in previous_wrong)
    native_known = [row for row in details if row['domain'] == 'new_native' and row['view'] == 'native' and row['family'] != 'Roboto']
    return {
        'roboto': {'views': len(roboto), 'baseline_wrongly_named': len(previous_wrong),
                   'additional_wrong_names_prevented': removed,
                   'additional_prevention_rate_among_baseline_wrong_names': removed / len(previous_wrong) if previous_wrong else None,
                   'baseline_already_not_wrongly_named': len(roboto) - len(previous_wrong),
                   'verifier_correct_and_above_gate': sum(bool(row['roboto_veto']) for row in roboto),
                   'note': 'Frozen roboto_veto_recall measures the verifier alone on all Roboto views; it is not incremental rejection recall.'},
        'new_native_known_clean': {family: {
            'views': sum(row['family'] == family for row in native_known),
            **{key: sum(bool(row[key]) for row in native_known if row['family'] == family)
               for key in ('before_correct', 'after_correct', 'before_wrong', 'after_wrong')},
        } for family in sorted({row['family'] for row in native_known})},
        'view_dependence': 'Four correlated views inherit each native source. View counts are not independent screenshot counts.'}


def main(args):
    from prepare_sans_views import load_views
    from flux_glyph.region_font import RegionFontClassifier
    require(not args.output.exists(), 'retain the existing test report')
    selection,parity=validate(args)
    policy=effective_policy(selection)
    evaluation_sha=sha(__file__)
    model=RegionFontClassifier(args.region)
    require(model.verifier_meta['model']['sha256']==parity['model_sha256']
            and sha(args.region/'metadata.json')==parity['metadata_sha256'], 'test model differs from frozen export')
    data=load_views(args.data,'test')
    require(data['partition_manifest']['frozen_selection']['sha256']==sha(args.run/'SELECTION.json'), 'test was derived for another selection')
    require(data['families'] == selection['families'] == model.verifier_meta['families'], 'test data/model class order differs')
    base,primary_families=baseline_observations(data,args.primary)
    blocks=[]
    for start in range(0,len(data['tiles']),128):
        block=np.array(data['tiles'][start:start+128],copy=True)
        # baseline_observations is frozen with the trainer. Independently audit
        # outputs that its naming-only summary does not consume, so a runtime
        # rejection for invalid size output cannot become a reported acceptance.
        primary_logits, primary_size=model.session.run(['logits','log_em_ratio'],{'tiles':block})
        require(primary_logits.dtype == primary_size.dtype == np.float32
                and primary_logits.shape == (len(block), len(primary_families)) and primary_size.shape == (len(block),)
                and np.isfinite(primary_logits).all() and np.isfinite(primary_size).all() and np.all(np.abs(primary_size) <= 3),
                'primary output differs from runtime-valid test outputs')
        logits,unused_size=model.verifier_session.run(['logits','log_em_ratio'],{'tiles':block})
        require(logits.dtype == unused_size.dtype == np.float32
                and logits.shape == (len(block),len(data['families'])) and unused_size.shape == (len(block),)
                and np.isfinite(logits).all() and np.isfinite(unused_size).all(),'invalid verifier test outputs')
        blocks.append(logits)
    obs=observations(np.concatenate(blocks),data['rows'],policy['temperature'],policy['gates'])
    metrics,details=summarize_selection(selection,data['rows'],base,obs,data['families'],primary_families)
    require(validate(args) == (selection,parity) and sha(__file__) == evaluation_sha,
            'frozen selection/model/evaluation source changed during test')
    require(sha(args.data/'MANIFEST.json') == data['manifest_sha256']
            and sha(args.data/'test/MANIFEST.json') == data['partition_manifest_sha256']
            and all(sha(args.data/'test'/data['partition_manifest'][key]['path']) == data['partition_manifest'][key]['sha256']
                    for key in ('array','metadata')), 'test view files changed during evaluation')
    report={'schema':'flux-glyph-sans-verifier-test-v1','passed':metrics['passed'],'metrics':metrics,'rows':details,
            'selection_schema':selection['schema'],'effective_verifier_policy':policy,
            'objective_protocol':selection.get('objective_protocol'),
            'diagnostics':diagnostics(details),'runtime_output_validity_audited':True,
            'selection_sha256':sha(args.run/'SELECTION.json'),'parity_sha256':sha(args.run/'PARITY.json'),
            'data_manifest_sha256':data['manifest_sha256'],'test_partition_manifest_sha256':data['partition_manifest_sha256'],
            'metadata_sha256':sha(args.region/'metadata.json'),'source_sha256':evaluation_sha,
            'test_used_for_training_or_selection':False,
            'scope':'Fresh controlled iOS native nine-family scenes plus reused R17 native regression; four correlated views per source. No Android-native accuracy claim.'}
    dump(args.output,report)
    print(json.dumps({'passed':report['passed'],'metrics':metrics},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','primary','region','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
