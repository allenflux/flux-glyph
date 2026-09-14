"""Fail-closed unit tests for the spatial-rejection exporter contract."""
from copy import deepcopy
import builtins
import importlib.util

import pytest

from training import export_spatial_rejection as export


def loss_parts(weight=.25):
    return {'unknown_oe':.2,'unknown_oe_weighted':.05,'unknown_oe_rows':16,
        'mobile_ce':2.,'mobile_size':.5,'mobile_loss':1.05,
        'pair_loss':.8,'pair_weighted':.08,'pair_count':16,'pair_rows':32,
        'original_total':2.5,'repair_unknown_ce':1.2,'repair_kaiti_ce':.4,
        'repair_system_confusion':.8,'repair_total':.25*1.2+.4+weight*.8}


def guard(actual=26, passed=True):
    return {**export.ANDROID_GUARD,'actual':actual,'passed':passed}


def selection():
    return {'trial':'rejection-low','families':[*(f'f{i}' for i in range(24)),'__unknown__'],
        'training_protocol_sha256':'a'*64,'training_counts_sha256':'b'*64,
        'initializer_state_sha256':'c'*64,'inherited_spatial_state_sha256':'d'*64,
        'state_after_sha256':'e'*64,'data_manifest_sha256':'f'*64,'repair_confusion_weight':.25,
        'objective':{'all_parameters_trained':False},'selected':{'metrics':{},'retention_checks':[]},
        'android_system_guard_policy':deepcopy(export.ANDROID_GUARD),
        'android_system_guard':guard(),'android_system_guard_baseline':guard(27)}


def measured():
    return {'retention_checks':[{'passed':True} for _ in range(46)],'metrics':{}}


def added():
    return {'passed':True,'checks':[{'passed':True} for _ in range(7)]}


def test_module_import_does_not_import_torch(monkeypatch):
    path = export.Path(export.__file__)
    original = builtins.__import__
    def denied(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('top-level Torch import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins,'__import__',denied)
    spec=importlib.util.spec_from_file_location('_spatial_export_without_torch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.PARITY_SCHEMA == 'flux-glyph-spatial-rejection-onnx-parity-v1'


def test_exact_spatial_contract_constants():
    assert export.ARCHITECTURE == 'region-cnn64x256-unified-spatial4x16-v1'
    assert export.PARAMETERS == 3810234 and export.TRAINABLE_PARAMETERS == 3145985
    assert export.CAL_REGIONS == 18672 and export.CAL_TILES == 37834 and export.STEPS == 2400
    assert export.TRIALS == ('rejection-low','rejection-medium','rejection-high')
    assert export.REPAIR_WEIGHTS == {'rejection-low':.25,'rejection-medium':.5,'rejection-high':1.}


@pytest.mark.skipif(not all((export.GRID_ROOT/(trial+'.json')).is_file()
                            for trial in export.TRIALS),
                    reason='ignored local spatial-rejection preflight artifacts unavailable')
def test_all_real_preflights_and_complete_objective_validate_without_torch():
    for trial in export.TRIALS:
        plan=export.read(export.GRID_ROOT/(trial+'.json'))
        assert export.validate_objective(plan['objective']) is plan['objective']
        assert export.validate_preflight(plan['preflight'],trial) is plan['preflight']
    changed=deepcopy(plan['objective']);changed['mobile_weight']=.25
    with pytest.raises(ValueError): export.validate_objective(changed)


def test_loss_trace_reconstructs_original_and_repair_totals():
    parts=loss_parts(); assert export.validate_loss_trace(parts,.25) is parts
    pre={'passed':True,'steps':4,'traces':[],'frozen_parameters_unchanged':True,
        'folding_parity_passed':True,'original_objective_equality_passed':True,
        'counts':{'optimizer_steps':4,'replay_rows':384,'unknown_oe_rows':64,'focus_rows':128,
            'focus_teacher_eligible_rows':114,'mobile_rows':80,'mobile_unknown_rows':12,
            'pair_count':64,'pair_rows':128},
        'residual':{'trainable_parameter_count':3145985,'max_absolute_residual':.01,
            'horizontal_sum_max_error':1e-9,'horizontal_sum_limit':1e-7}}
    for step in range(1,5):
        p=loss_parts();pre['traces'].append({'step':step,'loss':p['original_total']+p['repair_total'],
                                            'gradient_norm':1.,'parts':p})
    assert export.validate_preflight(pre,'rejection-low') is pre


@pytest.mark.parametrize(('field','value'),[
    ('unknown_oe_weighted',.1),('mobile_loss',2.1),('pair_weighted',.16),
    ('repair_total',9.),('pair_count',15)])
def test_changed_loss_or_row_budget_is_rejected(field,value):
    parts=loss_parts();parts[field]=value
    with pytest.raises(ValueError): export.validate_loss_trace(parts,.25)


def test_first_passing_grid_selection_is_strict():
    base={'selected':'rejection-medium','selection_policy':export.SELECTION_POLICY,
          'results':{'rejection-low':{'calibration_passed':False},
                     'rejection-medium':{'calibration_passed':True}}}
    assert export._selected_trial(base)=='rejection-medium'
    for changed in (
        {**base,'results':{**base['results'],'rejection-low':{'calibration_passed':True}}},
        {**base,'results':{'rejection-medium':{'calibration_passed':True}}},
        {**base,'selected':'rejection-high'}):
        with pytest.raises(ValueError): export._selected_trial(changed)


def test_android_guard_is_separate_and_fail_closed():
    assert export.validate_android_system_guard(export.ANDROID_GUARD,guard(),guard(27))==guard()
    for policy,current,baseline in (({**export.ANDROID_GUARD,'maximum':28},guard(),guard(27)),
                                    (export.ANDROID_GUARD,guard(28,False),guard(27)),
                                    (export.ANDROID_GUARD,guard(),guard(26))):
        with pytest.raises(ValueError): export.validate_android_system_guard(policy,current,baseline)


def test_metadata_describes_folded_residual_and_remains_pre_dev():
    s=selection(); cal={'manifest':{'font_label_groups':{'known':list(range(24))}}}
    meta=export.metadata_for_export(s,cal,{'font_sources':{}},'1'*64,'2'*64,'3'*64,measured(),added())
    assert meta['schema']==export.METADATA_SCHEMA
    assert meta['algorithm']=='unified-region-cnn64x256-v1'
    assert meta['network_architecture']==export.ARCHITECTURE
    assert meta['training']['all_parameters_trained'] is False
    assert meta['training']['cnn_input_rows_per_step']==180
    assert meta['training']['parameter_count']==3810234
    assert meta['training']['data_manifest_sha256']=='f'*64
    assert meta['training']['ocr_text_used'] is False
    assert meta['validation']['promotion_allowed'] is False
    assert meta['stable_validation_passed'] is meta['test_passed'] is False


def test_parity_names_trial_and_proves_cal53_plus_android_guard():
    s=selection(); p=export.build_parity_report(s,'1'*64,'2'*64,'3'*64,'4'*64,
        {'source':'sha'},{'cal':'sha'},{'logits':1e-5,'size':1e-5},.01,
        [{'batch_size':n,'samples':256,'passed':True,'font_and_size_checked':True}
         for n in (1,7,32,128)],measured(),added(),guard())
    assert p['schema']==export.PARITY_SCHEMA and p['trial']=='rejection-low'
    assert p['calibration_checks_passed']==53 and p['android_system_guard']['actual']==26
    assert p['parameter_count']==3810234 and p['one_deployed_cnn'] is True
    assert p['platform_routing'] is p['promotion_allowed'] is p['development_evaluated'] is False


class _Value:
    def __init__(self,name,shape,type_='tensor(float)'):
        self.name=name;self.shape=shape;self.type=type_


class _Session:
    def get_inputs(self): return [_Value('tiles',['batch',1,64,256])]
    def get_outputs(self): return [_Value('logits',['batch',25]),_Value('log_em_ratio',['batch'])]


class _Model:
    class Opset:
        version=17
    opset_import=[Opset()]


def test_onnx_contract_requires_opset17_and_batch_only_dynamic():
    export._validate_onnx_contract(_Model(),_Session())
    bad=_Session();bad.get_inputs=lambda:[_Value('tiles',['batch',1,'height',256])]
    with pytest.raises(ValueError): export._validate_onnx_contract(_Model(),bad)
