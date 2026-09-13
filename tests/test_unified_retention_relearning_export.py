"""Portable exporter checks for asymmetric teacher-preservation provenance."""
from copy import deepcopy

import pytest

from training import export_unified_retention_relearning as module
from training import train_unified_retention_relearning as trainer
from training.retention_asymmetric_teacher_loss import ASYMMETRIC_PRESERVATION
from test_unified_retention_margin_export import selection_fixture as margin_selection


def selection_fixture():
    selection=deepcopy(margin_selection())
    selection.update(schema='flux-glyph-unified-retention-wide-named-relearning-selection-v1',
        objective_variant=trainer.OBJECTIVE_VARIANT,objective=deepcopy(trainer.OBJECTIVE),
        design_changes=deepcopy(trainer.DESIGN_CHANGES),
        fixed_runtime=deepcopy(trainer.FIXED_RUNTIME),
        asymmetric_teacher_preservation=deepcopy(ASYMMETRIC_PRESERVATION),
        teacher_distribution_kl=False,teacher_distribution_kl_all_eligible_rows=False,
        teacher_distribution_kl_scope=ASYMMETRIC_PRESERVATION['distribution_kl_scope'],
        named_teacher_distribution_kl=False,unknown_teacher_distribution_kl=True,
        named_teacher_confidence_floor=True)
    return selection


def runtime_record():
    return {**deepcopy(module.FIXED_RUNTIME),'promotion_allowed':True,
        'retention_checks':[],'retention_populations':{},
        'metrics':{'passed':True,'named_precision':1.,'known_correct_coverage':1.,
            'unknown_not_named_rate':1.,'checks':{}}}


def parity(selection):
    return module.build_parity_report(selection,'a'*64,'b'*64,'c'*64,'d'*64,
        {'exporter':'e'*64},{'selection':'f'*64},{'logits':0.,'size':0.},0.,
        46,137,[{'batch_size':1,'passed':True}],{'metrics':{'passed':True}})


def test_actual_metadata_and_parity_builders_preserve_both_teacher_branches():
    selection=selection_fixture();metadata=module.training_metadata(selection);report=parity(selection)
    exported=module.metadata_for_export(selection,
        {'manifest':{'font_label_groups':{}}},{'font_sources':{}},
        'c'*64,'b'*64,'a'*64,runtime_record())
    for value in (metadata,report,exported['training']):
        assert value['asymmetric_teacher_preservation']==ASYMMETRIC_PRESERVATION
        assert value['objective']['asymmetric_teacher_preservation']==ASYMMETRIC_PRESERVATION
        assert value['teacher_distribution_kl'] is False
        assert value['teacher_distribution_kl_all_eligible_rows'] is False
        assert value['named_teacher_distribution_kl'] is False
        assert value['unknown_teacher_distribution_kl'] is True
        assert value['named_teacher_confidence_floor'] is True
        assert value['teacher_distribution_kl_scope']==ASYMMETRIC_PRESERVATION['distribution_kl_scope']
        assert value['angular_margin']==trainer.ANGULAR_MARGIN
    assert metadata['named_teacher_cache_used_partitions']==['original','supplement']
    assert metadata['unknown_teacher_used_partitions']==['original','supplement']
    assert metadata['teacher_mix']['known']['named_teacher_tiles']==4820
    assert metadata['teacher_mix']['known']['unknown_teacher_tiles']==0
    assert exported['validation']['kind']=='full_cnn_training_named_relearning_calibration_only_at_export'
    assert exported['model']=={'path':'model.onnx','sha256':'c'*64}
    assert exported['network_architecture']==trainer.ARCHITECTURE
    assert report['export_parameters_unchanged'] is True


@pytest.mark.parametrize('fault',(
    'spec','objective','global_kl','all_kl','named_kl','unknown_kl','named_floor',
    'scope','threshold','parameter','inference','named_source','unknown_source','known_teacher'))
def test_builders_reject_preservation_semantic_or_teacher_source_drift(fault):
    selection=selection_fixture()
    if fault=='spec':selection['asymmetric_teacher_preservation']['unknown_distribution_kl']=False
    elif fault=='objective':selection['objective']['named_teacher_confidence_floor']=False
    elif fault=='global_kl':selection['teacher_distribution_kl']=True
    elif fault=='all_kl':selection['teacher_distribution_kl_all_eligible_rows']=True
    elif fault=='named_kl':selection['named_teacher_distribution_kl']=True
    elif fault=='unknown_kl':selection['unknown_teacher_distribution_kl']=False
    elif fault=='named_floor':selection['named_teacher_confidence_floor']=False
    elif fault=='scope':selection['teacher_distribution_kl_scope']='all eligible rows'
    elif fault=='threshold':selection['asymmetric_teacher_preservation']['confidence_threshold']=.7
    elif fault=='parameter':selection['asymmetric_teacher_preservation']['adds_parameters']=True
    elif fault=='inference':selection['asymmetric_teacher_preservation']['inference_rule']=True
    elif fault=='named_source':selection['named_teacher_cache_used_partitions'].append('known')
    elif fault=='unknown_source':selection['unknown_teacher_used_partitions'].append('known')
    else:selection['teacher_mix']['known']['unknown_teacher_tiles']=1
    with pytest.raises(ValueError):module.training_metadata(selection)
    with pytest.raises(ValueError):parity(selection)


def test_fixed_model_step_runtime_and_replay_quota_remain_candidate18_values():
    selection=selection_fixture()
    assert module.validate_angular_margin_provenance(selection)==trainer.ANGULAR_MARGIN
    assert module.validate_asymmetric_teacher_provenance(selection)==ASYMMETRIC_PRESERVATION
    assert selection['steps']==selection['eval_every']==selection['optimizer_steps_executed']==6000
    assert selection['selected']['step']==6000
    assert selection['model_count']==selection['encoder_count']==1
    assert selection['fixed_runtime']==module.FIXED_RUNTIME
    expected=module.expected_family_counts(selection['families'],trainer.SAMPLING,6000)
    assert sum(expected.values())==576000
    assert expected['PingFang']==114000
    assert expected['SF Pro']==expected['Helvetica']==42000
    assert expected['Alipay Number']==18000
