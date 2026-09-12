"""Run with the optional standalone torch environment using unittest."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
try:
    import torch
    from network import FAMILIES, FontClassifier
    from region_network import RegionFontClassifier
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional PyTorch training environment required')
class RegionNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'parent.pth'
        self.names = FAMILIES[::-1]
        self.parent = FontClassifier()
        with torch.no_grad():
            for index in range(len(self.names)):
                self.parent.family_head.weight[index].fill_(index)
                self.parent.family_head.bias[index] = -index
        self.state = self.parent.state_dict()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        torch.save({'state_dict': self.state, 'families': self.names}, self.path)

    def test_named_head_remapping_preserves_trunk(self):
        self.save()
        families = ['PingFang SC', 'SF Pro']
        model = RegionFontClassifier(2)
        model.warm_start(self.path, families)
        for index, family in enumerate(families):
            expected = self.names.index(family)
            self.assertTrue(torch.equal(model.family_head.weight[index], torch.full((128,), float(expected))))
            self.assertEqual(float(model.family_head.bias[index].detach()), -expected)
        self.assertTrue(torch.equal(model.trunk[0].weight, self.parent.trunk[0].weight))
        self.assertEqual(int(torch.count_nonzero(model.size_head.weight)), 0)

    def test_missing_parent_trunk_is_rejected(self):
        del self.state['trunk.0.weight']
        self.save()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            RegionFontClassifier(2).warm_start(self.path, ['PingFang SC', 'SF Pro'])

    def test_untrained_family_is_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'cannot cover'):
            RegionFontClassifier(2).warm_start(self.path, ['PingFang SC', 'Untrained Font'])

    def test_rectangular_input_and_size_gradient(self):
        model = RegionFontClassifier(8)
        logits, ratio = model(torch.rand(2, 1, 64, 256))
        self.assertEqual(tuple(logits.shape), (2, 8))
        self.assertEqual(tuple(ratio.shape), (2,))
        self.assertTrue(torch.equal(ratio, torch.zeros(2)))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 7])) + ((ratio - .2) ** 2).mean()
        loss.backward()
        self.assertGreater(float(model.size_head.weight.grad.norm()), 0)
        self.assertGreater(float(model.family_head.weight.grad.norm()), 0)
        self.assertGreater(float(model.trunk[0].weight.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
