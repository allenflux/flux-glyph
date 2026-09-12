"""Torch-only lowering tests; formal ONNX parity remains a separate run."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional training torch environment")
class StableExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(37)

    def test_centered_double_retains_affine_parameters_and_original_groupnorm_definition(self):
        from export_region_stable import CenteredGroupNorm
        original = torch.nn.GroupNorm(8, 32).eval()
        with torch.no_grad():
            original.weight.uniform_(.3, 1.7)
            original.bias.uniform_(-.2, .2)
        lowered = CenteredGroupNorm(original, high_precision=True)
        self.assertTrue(torch.equal(original.weight, lowered.weight))
        self.assertTrue(torch.equal(original.bias, lowered.bias))
        for value in (torch.randn(3, 32, 64, 256), torch.zeros(2, 32, 16, 64), torch.rand(1, 32, 8, 32) * .01):
            with torch.inference_mode():
                expected, actual = original(value), lowered(value)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_replacement_preserves_every_frozen_parameter_key_and_value(self):
        from export_region_stable import replace_groupnorm
        from region_network import RegionFontClassifier
        model = RegionFontClassifier(8).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertEqual(replace_groupnorm(model, high_precision=True), 4)
        self.assertEqual(set(before), set(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))

    def test_whole_dual_output_model_remains_close_to_original_groupnorm(self):
        import copy
        from export_region_stable import replace_groupnorm
        from region_network import RegionFontClassifier
        original = RegionFontClassifier(8).eval()
        lowered = copy.deepcopy(original)
        replace_groupnorm(lowered, high_precision=True)
        with torch.inference_mode():
            x = torch.rand(3, 1, 64, 256)
            expected, actual = original(x), lowered(x)
        self.assertEqual(tuple(actual[0].shape), (3, 8))
        self.assertEqual(tuple(actual[1].shape), (3,))
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, atol=2e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
