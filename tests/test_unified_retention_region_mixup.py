"""Full-region and mixed true-label accounting remains auditable."""
from copy import deepcopy
import inspect
import pytest
from training import train_unified_retention_region_mixup as module
from test_unified_retention import FAMILIES,sampler_data


def counted_batch():
    rows,pools=sampler_data()
    rows=module.RetentionSampler(rows,FAMILIES,2026091406,pools).batch()
    mask=[r['family'] not in (*module.core.CORE_FAMILIES,module.core.UNKNOWN) for r in rows]
    weights=[2. if r['family']==module.core.UNKNOWN or (r['domain']=='ios' and r['view']=='native'
        and r['family'] in module.core.CORE_FAMILIES) else 1. for r in rows]
    counts=module.MixupCounts()
    tiles=[1+i%3 for i in range(96)]
    permutation=list(reversed(range(96)))
    counts.update(rows,mask,weights,tiles,.25,permutation)
    return counts,rows,mask,weights,tiles,permutation


def test_region_and_pair_totals_count_every_tile_and_unknown_on_both_sides():
    counts,rows,_,_,_,_=counted_batch();report=module.validate_mixup_counts(counts.report(),1)
    assert report['full_region_tile_histogram']=={'1':32,'2':32,'3':32}
    assert report['full_region_tiles']==192 and report['mixup_examples']==96
    assert report['mixup_lambda_sum']==.25 and report['mixup_lambda_square_sum']==.0625
    unknown=sum(r['family']==module.core.UNKNOWN for r in rows)
    pairs=report['mixup_pairs_by_unknown_count']
    assert pairs.get('1',0)+2*pairs.get('2',0)==2*unknown
    assert report['mixup_self_pairs']==0


@pytest.mark.parametrize('fault',['tile_count','pair_count','lambda','unknown_kl','core_kl','source_total','holdout','fake_unknown'])
def test_inconsistent_training_evidence_rejected(fault):
    counts,*_=counted_batch();report=deepcopy(counts.report())
    if fault=='tile_count':report['full_region_tiles']+=1
    elif fault=='pair_count':report['mixup_pairs_by_unknown_count']['0']+=1
    elif fault=='lambda':report['mixup_lambda_sum']=float('nan')
    elif fault in ('unknown_kl','core_kl'):
        family=module.core.UNKNOWN if fault=='unknown_kl' else 'PingFang'
        next(r for r in report['source_target_rows'] if r['target_family']==family)['base_kl_rows']=1
    elif fault=='source_total':report['source_target_rows'][0]['rows']+=1
    elif fault=='holdout':report['development_holdout_read']=True
    else:report['synthetic_unknown_labels']=True
    with pytest.raises(ValueError):module.validate_mixup_counts(report,1)


def test_unknown_base_kl_rejected_at_collection():
    _,rows,mask,weights,tiles,permutation=counted_batch()
    mask[next(i for i,r in enumerate(rows) if r['family']==module.core.UNKNOWN)]=True
    with pytest.raises(ValueError):module.MixupCounts().update(rows,mask,weights,tiles,.5,permutation)


def test_fixed_cache_reuse_preserves_original_feature_identity_and_no_new_extraction():
    args=module.parser().parse_args([])
    assert args.seed==2026091406 and args.steps==2000
    assert args.cache.name=='cache' and args.cache.parent.name=='run-adapter-v1'
    assert module.sha(args.cache/'CACHE_MANIFEST.json')==module.CACHE_MANIFEST_SHA
    source=inspect.getsource(module.train)
    assert 'extract_cache(' not in source
    assert "cached_identity['bindings']" in source
    assert "'cached_feature_extraction_performed':False" in source
    assert source.index("dump(args.output/'TRAINING_FREEZE.json'")<source.index('optimizer=torch.optim.AdamW')
