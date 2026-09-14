"""Verify the added loss is trained alongside the unchanged R22/short objective."""
import copy
from types import SimpleNamespace

import pytest


def test_v2_preserves_fixed_training_and_acceptance_contract():
    from training import train_unified_short_regions as v1
    from training import train_unified_short_regions_v2 as v2
    args=SimpleNamespace(teacher='teacher',short='short')
    old=v1.design(args,{})
    new=v2.design(args,{})
    for key in old:
        if key not in ('schema','objective'):
            assert old[key]==new[key]
    for key,value in old['objective'].items():
        assert new['objective'][key]==value
    assert new['objective']['original_full_correct_teacher_kl_preserved'] is True
    assert new['objective']['additional_confidence_retention']['runtime_changes'] is False
    assert new['schema']=='flux-glyph-unified-short-training-plan-v2'


def test_real_combined_v2_objective_adds_retention_and_updates_all_network_groups():
    torch=pytest.importorskip('torch')
    from training import train_unified_short_regions as v1
    from training import train_unified_short_regions_v2 as v2
    from training.short_confidence_retention import confidence_retention_loss
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4);torch.manual_seed(13)
    model=WideRegionFontClassifier(25).train()
    targets=torch.tensor([i%25 for i in range(96)])
    small_targets=torch.tensor(list(range(24))+[24]*8)
    rows=[{'split':'train','native_font_verified':True,'target':int(t),'family':v2.FAMILIES[int(t)],
           'source_font_family':v2.FAMILIES[int(t)],'font_face':'synthetic',
           'view':'native','domain':'android','tile_count':1 if i%2 else 2}
          for i,t in enumerate(targets)]
    small=[{'split':'train','native_font_verified':True,'target':int(t),
            'family':v2.FAMILIES[int(t)],'glyph_count':2} for t in small_targets]
    teacher=torch.zeros(96,25);teacher[torch.arange(96),targets]=4.
    batch={'images':torch.rand(128,1,64,256),'teacher':teacher,'targets':targets,
           'sizes':torch.linspace(-.2,.2,96),'rows':rows,'short_targets':small_targets,
           'short_sizes':torch.linspace(-.15,.15,32),'short_rows':small}
    old,old_parts=v1.objective(model,batch)
    new,new_parts=v2.objective(model,batch)
    logits,_,_=v2.forward_with_features(model,batch['images'])
    extra,eligible,boosted=confidence_retention_loss(logits[:96],teacher,targets,rows)
    assert extra>0 and eligible.all() and boosted.any()
    assert torch.allclose(new,old+extra)
    for key,value in old_parts.items():assert torch.allclose(value,new_parts[key])
    assert new_parts['confidence_eligible']==96
    assert new_parts['confidence_boosted']==boosted.sum()
    new.backward()
    for group in ('trunk','style','family_head','size_head'):
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(group+'.')]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool(g.abs().sum()>0) for g in grads)


def test_v2_correct_teacher_loss_never_changes_labels_or_runtime():
    from training import train_unified_short_regions as v1
    from training import train_unified_short_regions_v2 as v2
    assert v2.FAMILIES==v1.FAMILIES
    assert v2.FIXED_RUNTIME==v1.FIXED_RUNTIME
    assert v2.BASE_SHA==v1.BASE_SHA
    assert v2.EXTRA_ACCEPTANCE==v1.EXTRA_ACCEPTANCE
    assert v2.OBJECTIVE['short_teacher'] is None
    assert v2.OBJECTIVE['all_parameters_trained'] is True


def test_v2_cal_pass_still_requires_export_and_development(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    import importlib.util
    from pathlib import Path
    from training import train_unified_short_regions_v2 as v2
    path=Path(__file__).with_name('test_unified_short_training.py')
    spec=importlib.util.spec_from_file_location('short_training_gate_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture,'trainer',v2)
    fixture.test_calibration_success_is_recorded_but_never_becomes_release_promotion(monkeypatch,tmp_path)
