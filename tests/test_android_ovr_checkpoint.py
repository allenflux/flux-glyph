"""Exact OVR inheritance checks on tiny CPU-only parameter fixtures."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from export_android_ovr import validate_checkpoint
from train_regions import sha,state_sha
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch is only installed in the separate training environment')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        from android_ovr_network import AndroidOVRClassifier,ARCHITECTURE
        from region_network import RegionFontClassifier
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        families=[f'Family {index}' for index in range(9)]+['__unknown__']
        parent=RegionFontClassifier(10).eval()
        state=AndroidOVRClassifier(10,9).from_parent(parent.state_dict()).state_dict()
        initial={key:value.clone() for key,value in state.items() if key.startswith('heads.')}
        initial_path=self.root/'INITIAL_HEADS.pth';torch.save(initial,initial_path)
        before=state_sha(state)
        state['heads.0.0.weight'].view(-1)[0]+=.1
        path=self.root/'parent.pth';torch.save({'families':families,'state_dict':parent.state_dict()},path)
        frozen=state_sha({key:value for key,value in state.items() if not key.startswith('heads.')})
        self.selection={'families':families,'parent_checkpoint':{'path':str(path),'sha256':sha(path)},
            'parent_frozen_state_sha256':frozen,
            'bindings':{str(initial_path):sha(initial_path)},'heads_before_sha256':state_sha(initial),'state_before_sha256':before,
            'heads_after_sha256':state_sha({key:value for key,value in state.items() if key.startswith('heads.')})}
        self.checkpoint={'architecture':ARCHITECTURE,'unknown_index':9,'state_dict':state,
            'parent_checkpoint_sha256':sha(path),'parent_frozen_state_sha256':frozen}

    def tearDown(self):self.temp.cleanup()

    def test_exact_parent_and_bound_heads_are_accepted(self):
        validate_checkpoint(self.checkpoint,self.selection,self.root)

    def test_encoder_size_or_head_parameter_changes_are_rejected(self):
        encoder=next(key for key in self.checkpoint['state_dict'] if key.startswith('trunk.') and key.endswith('weight'))
        for key in (encoder,'size_head.bias','heads.0.0.weight'):
            bad=copy.deepcopy(self.checkpoint);bad['state_dict'][key].view(-1)[0]+=.1
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_checkpoint(bad,self.selection,self.root)

    def test_wrong_unknown_order_or_parent_file_is_rejected(self):
        bad=copy.deepcopy(self.checkpoint);bad['unknown_index']=0
        with self.assertRaises(ValueError):validate_checkpoint(bad,self.selection,self.root)
        path=Path(self.selection['parent_checkpoint']['path']);path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):validate_checkpoint(self.checkpoint,self.selection,self.root)

    def test_nonfinite_parameters_are_rejected(self):
        self.checkpoint['state_dict']['heads.0.0.weight'].view(-1)[0]=float('nan')
        with self.assertRaises(ValueError):validate_checkpoint(self.checkpoint,self.selection,self.root)

    def test_initial_head_file_cannot_be_swapped_after_training(self):
        path=self.root/'INITIAL_HEADS.pth';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):validate_checkpoint(self.checkpoint,self.selection,self.root)


if __name__=='__main__':unittest.main()
