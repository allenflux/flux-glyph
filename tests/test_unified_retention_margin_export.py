"""Portable export-metadata checks for the TRAIN-only angular margin."""
from copy import deepcopy

import pytest

from training import export_unified_retention_margin as module
from training import train_unified_retention_margin as trainer
from training.retention_angular_margin_loss import ANGULAR_MARGIN
from test_unified_retention_face_balanced_export import selection_fixture as face_balanced_selection


def selection_fixture():
    selection=deepcopy(face_balanced_selection())
    selection.update(schema='flux-glyph-unified-retention-wide-angular-margin-selection-v1',
        architecture=trainer.ARCHITECTURE,objective_variant=trainer.OBJECTIVE_VARIANT,
        objective=deepcopy(trainer.OBJECTIVE),angular_margin=deepcopy(ANGULAR_MARGIN),
        angular_margin_applies_at_inference=False,
        angular_margin_adds_deployment_parameters=False,
        design_changes=deepcopy(trainer.DESIGN_CHANGES),steps=trainer.STEPS,
        eval_every=trainer.EVAL_EVERY,model_count=1,encoder_count=1,
        platform_routing=False,score_merging=False,training_inputs=['image_tiles'])
    return selection


def runtime_record():
    return {**deepcopy(module.FIXED_RUNTIME),'promotion_allowed':True,
        'retention_checks':[],'retention_populations':{},
        'metrics':{'passed':True,'named_precision':1.,'known_correct_coverage':1.,
            'unknown_not_named_rate':1.,'checks':{}}}


def build_report(selection):
    return module.build_parity_report(selection,'a'*64,'b'*64,'c'*64,'d'*64,
        {'exporter':'e'*64},{'selection':'f'*64},{'logits':0.,'size':0.},0.,
        46,137,[{'batch_size':1,'passed':True}],{'metrics':{'passed':True}})


def test_actual_builders_preserve_canonical_auxiliary_and_compact_teacher_handoff():
    selection=selection_fixture()
    metadata=module.training_metadata(selection)
    report=build_report(selection)
    exported=module.metadata_for_export(selection,
        {'manifest':{'font_label_groups':{}}},{'font_sources':{}},
        'c'*64,'b'*64,'a'*64,runtime_record())
    for value in (metadata,report,exported['training']):
        assert value['angular_margin']==ANGULAR_MARGIN
        assert value['objective']['angular_margin']==ANGULAR_MARGIN
        assert value['angular_margin_applies_at_inference'] is False
        assert value['angular_margin_adds_deployment_parameters'] is False
        assert value['known_cache']==selection['known_cache']
        assert value['merged_known_teacher_cache']==selection['known_cache']
        assert 'bindings' not in value['known_cache']
    assert exported['validation']['kind']=='full_cnn_training_angular_margin_calibration_only_at_export'
    assert metadata['selected_step']==exported['training']['selected_step']==trainer.STEPS
    assert report['frozen_retention_summary']['step']==trainer.STEPS
    assert exported['model']=={'path':'model.onnx','sha256':'c'*64}
    assert exported['network_architecture']==trainer.ARCHITECTURE
    assert exported['training']['model_count']==exported['training']['encoder_count']==1
    assert exported['training']['training_inputs']==['image_tiles']
    assert report['export_parameters_unchanged'] is True
    assert report['font_model_count']==report['encoder_count']==1


@pytest.mark.parametrize('fault',(
    'margin','objective','inference','deployment','step','encoder','input'))
def test_builders_reject_auxiliary_or_raw_runtime_contract_drift(fault):
    selection=selection_fixture()
    if fault=='margin':selection['angular_margin']['scale']+=1
    elif fault=='objective':selection['objective']['angular_margin']['monotone_guard']=False
    elif fault=='inference':selection['angular_margin_applies_at_inference']=True
    elif fault=='deployment':selection['angular_margin_adds_deployment_parameters']=True
    elif fault=='step':selection['selected']['step']-=1
    elif fault=='encoder':selection['encoder_count']=2
    else:selection['training_inputs']=['image_tiles','angular_features']
    with pytest.raises(ValueError):module.training_metadata(selection)
    with pytest.raises(ValueError):build_report(selection)


def test_fixed_final_contract_and_family_quota_math_are_unchanged():
    selection=selection_fixture()
    assert module.validate_angular_margin_provenance(selection)==ANGULAR_MARGIN
    assert selection['steps']==selection['eval_every']==selection['optimizer_steps_executed']==trainer.STEPS
    expected=module.expected_family_counts(selection['families'],trainer.SAMPLING,trainer.STEPS)
    assert sum(expected.values())==trainer.BATCH_SIZE*trainer.STEPS==576000
    assert expected['PingFang']==114000
    assert expected['SF Pro']==expected['Helvetica']==42000
    assert expected['Alipay Number']==18000
