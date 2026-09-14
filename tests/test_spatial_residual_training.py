import numpy as np
import pytest

torch = pytest.importorskip('torch')
from train_spatial_residual import feature_batch


def test_cached_features_keep_original_replay_focus_mobile_pair_order_and_targets():
    arrays = {name: np.tile(np.array([number, number+.5], np.float32)[:,None], (1,8192))
              for name,number in [('original',1),('supplement',2),('known',3),('ios',4),('android',5)]}
    raw = {'images':torch.zeros(180,1,64,256), 'indices':[('original',0)]*96,
           'focus_indices':[('known',1)]*32,
           'mobile_rows':[{'short_dataset':'ios','tile_start':0}]*20,
           'pair_rows':[{'short_dataset':'android','tile_start':1}]*32,
           'targets':torch.arange(96), 'teacher':torch.rand(96,25), 'rows':['unchanged']}
    result = feature_batch(raw, arrays, 'cpu')
    expected = torch.tensor([1.]*96+[3.5]*32+[4.]*20+[5.5]*32)
    torch.testing.assert_close(result['images'][:,0], expected)
    assert result['images'].shape == (180,8192)
    assert raw['images'].shape == (180,1,64,256)
    assert result['rows'] is raw['rows']
    torch.testing.assert_close(result['targets'],raw['targets'])
    torch.testing.assert_close(result['teacher'],raw['teacher'])
    arrays['known'][1,0] = np.nan
    with pytest.raises(ValueError, match='Invalid frozen features'):
        feature_batch(raw,arrays,'cpu')
