"""OVR training tests use artificial metadata/tensors and never load TEST data."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from training import train_android_ovr as ovr
from test_android_balanced import FAMILIES,synthetic_rows


def cache_fixture(tmp_path):
    source=tmp_path/'source.py';source.write_text('frozen source fixture')
    datasets={}
    for split in ('train','calibration'):
        datasets[split]={'rows':[{'split':split}],'tiles':np.zeros((2,1,64,256),np.float32),
            'families':FAMILIES,'manifest_sha256':'d'*64,'partition_sha256':split+'-sha',
            'partition':{'split':split,'array':{'path':'tiles.raw','sha256':'t'*64},
                         'metadata':{'path':'rows.json','sha256':'r'*64}}}
    identity=ovr.cache_identity(datasets,{'path':'parent.pth','sha256':'p'*64},'f'*64,
                                {str(source):ovr.sha(source)})
    root=tmp_path/'cache';root.mkdir();partitions={}
    for split in ('train','calibration'):
        (root/split).mkdir();partitions[split]={}
        for name,shape in (('features',(2,128)),('log_em_ratio',(2,))):
            path=root/split/(name+'.npy');np.save(path,np.zeros(shape,np.float32))
            partitions[split][name]={'path':f'{split}/{name}.npy','shape':list(shape),
                'dtype':'float32','sha256':ovr.sha(path)}
    ovr.dump(root/'PLAN.json',{'identity':identity,'device':'cpu'})
    ovr.dump(root/'MANIFEST.json',{'schema':'flux-glyph-android-ovr-embedding-cache-v1',
        'identity':identity,'partitions':partitions,'extraction_plan_sha256':ovr.sha(root/'PLAN.json')})
    return root,identity,datasets,source


def test_cache_reuses_exact_bound_train_cal_arrays(tmp_path):
    root,identity,_,_=cache_fixture(tmp_path)
    loaded,_=ovr.load_cache(root,identity)
    assert set(loaded)=={'train','calibration'}
    assert isinstance(loaded['train']['features'],np.memmap)
    assert loaded['train']['features'].shape==(2,128)
    assert not (root/'test').exists()


@pytest.mark.parametrize('problem',['source','array','plan','parent','dataset','test','path','shape','nan','size'])
def test_cache_fails_closed_on_identity_or_bytes_tampering(tmp_path,problem):
    root,identity,_,source=cache_fixture(tmp_path)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if problem=='source':source.write_text('source changed')
    elif problem=='array':
        with (root/'train/features.npy').open('ab') as stream:stream.write(b'tampered')
    elif problem=='plan':
        plan=json.loads((root/'PLAN.json').read_text());plan['device']='mps';ovr.dump(root/'PLAN.json',plan)
    elif problem=='parent':identity['parent_checkpoint']['sha256']='other'
    elif problem=='dataset':identity['data_manifest_sha256']='other'
    elif problem=='test':manifest['partitions']['test']=manifest['partitions']['train']
    elif problem=='path':manifest['partitions']['train']['features']['path']='../features.npy'
    elif problem=='shape':manifest['partitions']['train']['features']['shape']=[1,128]
    elif problem in ('nan','size'):
        name='features' if problem=='nan' else 'log_em_ratio'
        path=root/'train'/(name+'.npy');array=np.load(path);array.flat[0]=np.nan if problem=='nan' else 4.
        np.save(path,array);manifest['partitions']['train'][name]['sha256']=ovr.sha(path)
    ovr.dump(root/'MANIFEST.json',manifest)
    with pytest.raises(ValueError):ovr.load_cache(root,identity)


def test_cache_identity_rejects_test_inputs_and_wrong_split_rows(tmp_path):
    _,identity,datasets,_=cache_fixture(tmp_path)
    args=(identity['parent_checkpoint'],identity['parent_frozen_state_sha256'],identity['bindings'])
    extra=deepcopy(datasets);extra['test']=deepcopy(datasets['train'])
    with pytest.raises(ValueError,match='only accepts TRAIN and CAL'):ovr.cache_identity(extra,*args)
    datasets['train']['rows'][0]['split']='test'
    with pytest.raises(ValueError,match='split metadata differs'):ovr.cache_identity(datasets,*args)


def test_ovr_batches_keep_known_classes_unknown_families_and_faces_balanced():
    sampler=ovr.BalancedSampler(synthetic_rows(),FAMILIES,23)
    for _ in range(90):
        rows=ovr.balanced_batch(sampler)
        assert len(rows)==128 and sum(row['target']==9 for row in rows)==64
    report=sampler.report()
    assert set(report['known_class_rows'].values())=={640}
    assert set(report['unknown_source_family_rows'].values())=={1920}
    assert [row['rows'] for row in report['unknown_source_face_rows'] if row['family']=='Unknown C']==[480]*4


def test_ovr_loss_weights_and_gradients_match_independent_binary_evidence():
    torch=pytest.importorskip('torch')
    logits=torch.zeros((4,10),requires_grad=True);targets=torch.tensor([0,7,9,9])
    loss,parts=ovr.ovr_loss(logits,targets,9)
    assert loss.item()==pytest.approx(math.log(2),rel=1e-6)
    assert all(value.item()==pytest.approx(math.log(2),rel=1e-6) for value in parts.values())
    loss.backward()
    assert logits.grad[0,0].item()==pytest.approx(-.125)
    assert logits.grad[0,1].item()==pytest.approx(.25*.5/16)
    assert logits.grad[2,0].item()==pytest.approx(.25*.5/18)
    assert torch.count_nonzero(logits.grad[:,9])==0


def test_ovr_loss_handles_nonlast_unknown_and_rejects_one_sided_batches():
    torch=pytest.importorskip('torch')
    logits=torch.zeros((2,10),requires_grad=True);targets=torch.tensor([7,2])
    loss,_=ovr.ovr_loss(logits,targets,2);loss.backward()
    assert logits.grad[0,7]<0 and logits.grad[0,1]>0
    assert torch.count_nonzero(logits.grad[:,2])==0 and bool((logits.grad[1,[0,1,3,4,5,6,7,8,9]]>0).all())
    for targets in (torch.tensor([0,1]),torch.tensor([2,2]),torch.tensor([0,10])):
        with pytest.raises(ValueError):ovr.ovr_loss(logits,targets,2)


def test_head_optimizer_preserves_parent_encoder_and_size_byte_for_byte():
    torch=pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    from android_ovr_network import AndroidOVRClassifier
    torch.manual_seed(19);parent=RegionFontClassifier(10).state_dict()
    model=AndroidOVRClassifier(10,9).from_parent(parent)
    before=ovr.verify_inheritance(model,parent);heads_before=ovr.state_sha(ovr.head_state(model.state_dict()))
    optimizer=torch.optim.AdamW(model.heads.parameters(),lr=.001)
    logits=model.classify(torch.randn(128,128));targets=torch.tensor(list(range(9))*7+[0]+[9]*64)
    loss,_=ovr.ovr_loss(logits,targets,9);loss.backward();optimizer.step()
    assert ovr.verify_inheritance(model,parent)==before
    assert ovr.state_sha(ovr.head_state(model.state_dict()))!=heads_before
    assert all(value.grad is None for key,value in model.named_parameters() if not key.startswith('heads.'))
    with torch.no_grad():model.size_head.bias.add_(1)
    with pytest.raises(ValueError,match='encoder or size'):ovr.verify_inheritance(model,parent)


def test_protocol_defaults_keep_original_policy_and_fixed_4000_steps():
    from train_android_regions import POLICY
    args=ovr.parser().parse_args(['--data','data','--output','run'])
    assert args.steps==4000 and args.feature_device=='mps' and args.device=='cpu'
    assert ovr.POLICY==POLICY and ovr.LEARNING_RATE==.001 and ovr.OBJECTIVE['size_loss_weight']==0
    assert args.checkpoint.parent.name=='run-v2'
