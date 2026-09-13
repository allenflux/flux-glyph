from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    from android_ovr_network import AndroidOVRClassifier
    from region_network import RegionFontClassifier


@unittest.skipIf(torch is None, 'PyTorch is available in the local training environment')
class AndroidOVRNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1303)
        self.parent = RegionFontClassifier(10).eval()
        with torch.no_grad():
            self.parent.size_head.weight.normal_(0, .01)
            self.parent.size_head.bias.fill_(.25)

    def test_complete_parent_encoder_features_and_size_are_preserved(self):
        model = AndroidOVRClassifier().from_parent(self.parent.state_dict()).eval()
        expected = {key: value for key, value in self.parent.state_dict().items()
                    if not key.startswith('family_head.')}
        inherited = {key: value for key, value in model.state_dict().items()
                     if not key.startswith('heads.')}
        self.assertEqual(inherited.keys(), expected.keys())
        self.assertTrue(all(torch.equal(inherited[key], expected[key]) for key in inherited))
        self.assertEqual(sum(isinstance(module, torch.nn.GroupNorm) for module in model.modules()), 4)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                tiles = torch.rand(batch_size, 1, 64, 256)
                with torch.inference_mode():
                    expected_features = self.parent.style(self.parent.pool(self.parent.trunk(tiles)))
                    features = model.features(tiles)
                    logits, ratio = model(tiles)
                    _, expected_ratio = self.parent(tiles)
                self.assertEqual(features.shape, (batch_size, 128))
                self.assertEqual(logits.shape, (batch_size, 10))
                self.assertEqual(ratio.shape, (batch_size,))
                self.assertTrue(torch.equal(features, expected_features))
                self.assertTrue(torch.equal(ratio, expected_ratio))
                self.assertTrue(torch.equal(logits[:, 9], torch.zeros(batch_size)))

    def test_unknown_zero_and_independent_heads_preserve_family_order(self):
        for unknown_index in (0, 3, 9):
            with self.subTest(unknown_index=unknown_index):
                model = AndroidOVRClassifier(unknown_index=unknown_index)
                with torch.no_grad():
                    for index, head in enumerate(model.heads):
                        head[0].weight.zero_()
                        head[0].bias.zero_()
                        head[2].weight.zero_()
                        head[2].bias.fill_(index + 1)
                expected = list(range(1, 10))
                expected.insert(unknown_index, 0)
                self.assertTrue(torch.equal(model.classify(torch.randn(7, 128)),
                                            torch.tensor([expected] * 7, dtype=torch.float32)))
                self.assertEqual(len({head[0].weight.data_ptr() for head in model.heads}), 9)
                self.assertEqual(sum(parameter.numel() for parameter in model.heads.parameters()), 37449)

    def test_optimizer_updates_only_heads_and_train_keeps_parent_in_eval_mode(self):
        model = AndroidOVRClassifier().from_parent(self.parent.state_dict()).train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertTrue(model.training and model.heads.training)
        self.assertTrue(all(not module.training for module in
                            (model.trunk, model.pool, model.style, model.size_head)))
        self.assertTrue(all(parameter.requires_grad == name.startswith('heads.')
                            for name, parameter in model.named_parameters()))
        optimizer = torch.optim.SGD(model.heads.parameters(), lr=.1)
        logits, _ = model(torch.rand(4, 1, 64, 256))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 9]))
        loss.backward()
        optimizer.step()
        after = model.state_dict()
        self.assertTrue(all(torch.equal(before[key], value) for key, value in after.items()
                            if not key.startswith('heads.')))
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in after.items()
                            if key.startswith('heads.')))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if not name.startswith('heads.')))

    def test_invalid_parent_rejected_before_any_parameter_is_copied(self):
        for change in ('missing', 'extra', 'shape', 'dtype', 'nan', 'head_shape', 'head_nan', 'not_tensor'):
            with self.subTest(change=change):
                model = AndroidOVRClassifier()
                before = {key: value.clone() for key, value in model.state_dict().items()}
                state = {key: value.clone() for key, value in self.parent.state_dict().items()}
                key = 'trunk.0.weight'
                if change == 'missing':
                    del state['size_head.bias']
                elif change == 'extra':
                    state['unexpected'] = torch.zeros(1)
                elif change == 'shape':
                    state[key] = state[key][:1]
                elif change == 'dtype':
                    state[key] = state[key].double()
                elif change == 'nan':
                    state[key].flatten()[0] = float('nan')
                elif change == 'head_shape':
                    state['family_head.weight'] = state['family_head.weight'][:9]
                elif change == 'head_nan':
                    state['family_head.bias'][0] = float('nan')
                else:
                    state[key] = state[key].tolist()
                with self.assertRaisesRegex(ValueError, 'Parent state|Invalid parent tensor'):
                    model.from_parent(state)
                self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))

    def test_parent_tensors_not_shared_and_new_head_weights_preserved(self):
        model = AndroidOVRClassifier()
        original_heads = {key: value.clone() for key, value in model.heads.state_dict().items()}
        state = self.parent.state_dict()
        model.from_parent(state)
        inherited = model.trunk[0].weight.clone()
        state['trunk.0.weight'].add_(1)
        self.assertTrue(torch.equal(model.trunk[0].weight, inherited))
        self.assertTrue(all(torch.equal(original_heads[key], value)
                            for key, value in model.heads.state_dict().items()))

    def test_invalid_class_contract_rejected(self):
        for family_count, unknown_index in ((9, 8), (11, 9), (True, 9), (10, -1), (10, 10), (10, True)):
            with self.subTest(family_count=family_count, unknown_index=unknown_index):
                with self.assertRaisesRegex(ValueError, 'nine known families'):
                    AndroidOVRClassifier(family_count, unknown_index)


if __name__ == '__main__':
    unittest.main()
