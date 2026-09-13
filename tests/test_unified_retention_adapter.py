"""Residual-head losses and cached images retain the frozen single-CNN contract."""
from copy import deepcopy
import json
import numpy as np
import pytest
from training import train_unified_retention_adapter as module
from test_unified_retention import FAMILIES,sampler_data


def batch():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(14)
    rows,pools=sampler_data();rows=module.RetentionSampler(rows,FAMILIES,2026091405,pools).batch()
    targets=torch.tensor([r['target'] for r in rows]);base=torch.full((96,25),-2.,dtype=torch.float64)
    base[torch.arange(96),targets]=2.
    logits=torch.randn(96,25,dtype=torch.float64,requires_grad=True)
    return torch,rows,targets,base,logits


def test_residual_loss_native_and_unknown_weights_exact_gradient_and_no_size_term():
    torch,rows,targets,base,logits=batch()
    total,ce,kl,mask,weights=module.adapter_loss(logits,base,targets,rows,FAMILIES)
    expected_weights=torch.tensor([2. if r['family']==module.core.UNKNOWN or (r['domain']=='ios' and r['view']=='native'
        and r['family'] in module.core.CORE_FAMILIES) else 1. for r in rows],dtype=torch.float64)
    assert weights==expected_weights.tolist() and all(weights[i]==2. for i,r in enumerate(rows) if r['family']==module.core.UNKNOWN)
    smooth=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    torch.testing.assert_close(ce,(-(smooth*logits.log_softmax(1)).sum(1)*expected_weights).sum()/96)
    expected_mask=torch.tensor([r['family'] not in module.core.CORE_FAMILIES for r in rows])
    assert torch.equal(mask,expected_mask)
    probs=base[mask].softmax(1)
    torch.testing.assert_close(kl,(probs*(probs.log()-logits[mask].log_softmax(1))).sum(1).mean())
    torch.testing.assert_close(total,ce+kl)
    total.backward();gradient=(logits.detach().softmax(1)-smooth)*expected_weights[:,None]/96
    gradient[mask]+=(logits.detach()[mask].softmax(1)-probs)/mask.sum()
    torch.testing.assert_close(logits.grad,gradient)
    assert base.grad is None and module.OBJECTIVE['size_loss_weight']==0


def test_wrong_base_targets_are_excluded_and_empty_regularizer_differentiable():
    torch,rows,targets,base,logits=batch();base.fill_(-2.);base[torch.arange(96),(targets+1)%25]=2.
    total,ce,kl,mask,_=module.adapter_loss(logits,base,targets,rows,FAMILIES)
    assert not mask.any() and kl.item()==0 and kl.requires_grad
    total.backward();assert logits.grad.abs().sum()>0
    torch.testing.assert_close(total,ce)


@pytest.mark.parametrize('fault',['holdout','target','nan_base','base_grad'])
def test_invalid_training_identity_or_frozen_base_fails(fault):
    torch,rows,targets,base,logits=batch()
    if fault=='holdout':rows[0]={**rows[0],'split':'development_holdout'}
    elif fault=='target':targets[0]=(targets[0]+1)%25
    elif fault=='nan_base':base[0,0]=float('nan')
    else:base.requires_grad_(True)
    with pytest.raises(ValueError):module.adapter_loss(logits,base,targets,rows,FAMILIES)


def small_cache(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(2)
    from retention_adapter_network import RetentionAdapterClassifier
    model=RetentionAdapterClassifier(25)
    source=tmp_path/'source.txt';source.write_text('Frozen native fixture identity')
    datasets={}
    for split in ('train','calibration'):
        datasets[split]={'families':FAMILIES,'manifest_sha256':'a'*64,'partition_sha256':split,
            'rows':[{'split':split,'tile_start':0,'tile_count':2}],
            'partition':{'split':split,'metadata':{'path':'rows.json','sha256':split+'-rows'},
                         'array':{'path':'tiles.npy','sha256':split+'-tiles'}},
            'tiles':np.random.default_rng(17).normal(size=(2,1,64,256)).astype(np.float32)}
    identity=module.cache_identity(datasets,{'path':str(source),'sha256':module.sha(source)},
        module.state_sha(module.frozen_state(model.state_dict())),{str(source):module.sha(source)},'cpu')
    folder=tmp_path/'cache';cached,manifest=module.extract_cache(folder,model,datasets,identity,'cpu')
    return torch,model,datasets,identity,folder,cached,manifest


def test_bound_cache_and_complete_forward_have_equal_outputs_with_frozen_size(tmp_path):
    torch,model,data,identity,folder,cache,manifest=small_cache(tmp_path)
    assert (folder/'CACHE_FREEZE.json').is_file() and manifest['cache_freeze_sha256']==module.sha(folder/'CACHE_FREEZE.json')
    with torch.no_grad():
        model.residual_family_head[2].bias.fill_(.1)
        logits,size=model(torch.from_numpy(data['calibration']['tiles']))
    cached_logits,cached_size=module.infer_cached(model,cache['calibration'])
    np.testing.assert_array_equal(logits.numpy(),cached_logits)
    np.testing.assert_array_equal(size.numpy(),cached_size)
    loaded,_=module.load_cache(folder,identity)
    assert np.array_equal(loaded['train']['features'],cache['train']['features'])
    assert set(manifest['partitions'])=={'train','calibration'}


@pytest.mark.parametrize('fault',['array','rows','tiles','device','freeze','source'])
def test_cache_reuse_rejects_identity_or_array_tampering(tmp_path,fault):
    torch,model,data,identity,folder,cache,manifest=small_cache(tmp_path)
    identity=deepcopy(identity)
    if fault=='array':
        path=folder/'train/base_logits.npy';value=np.load(path).copy();value[0,0]+=1;np.save(path,value)
    elif fault=='rows':identity['partitions']['train']['rows']['sha256']='changed'
    elif fault=='tiles':identity['partitions']['train']['tiles']['sha256']='changed'
    elif fault=='device':identity['feature_device']='mps'
    elif fault=='freeze':(folder/'CACHE_FREEZE.json').write_text('{}')
    else:(tmp_path/'source.txt').write_text('changed')
    with pytest.raises(ValueError):module.load_cache(folder,identity)


def test_cache_identity_rejects_any_holdout_partition(tmp_path):
    torch,model,data,identity,folder,cache,manifest=small_cache(tmp_path)
    data['development_holdout']=data['calibration']
    with pytest.raises(ValueError):module.cache_identity(data,{},'a'*64,{},'cpu')


def test_fixed_optimizer_and_model_scope():
    args=module.parser().parse_args([])
    assert args.steps==2000 and args.seed==2026091405 and args.feature_device=='mps'
    assert module.EVAL_EVERY==200 and module.BATCH_SIZE==96 and module.LEARNING_RATE==5e-4 and module.MINIMUM_LEARNING_RATE==5e-5
    assert module.OBJECTIVE['base_frozen'] and module.OBJECTIVE['size_head_frozen']
    assert module.OBJECTIVE['external_teacher'] is False and module.OBJECTIVE['label_weights_are_inference_rules'] is False
