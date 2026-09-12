"""Guards for the binary labels, held-out families and pixel partition access."""
import json
from pathlib import Path
import tempfile
import unittest

from training.capture.prepare_unknown_regions import (
    KNOWN, SCHEMA, UNKNOWN_SPLITS, load_partition, preprocessing_contract, validate_family_label,
)


class UnknownCaptureDataTests(unittest.TestCase):
    def test_all_known_families_are_known_in_every_split(self):
        for split in UNKNOWN_SPLITS:
            for family in KNOWN + ['PingFang SC', 'PingFang TC', 'PingFang HK']:
                with self.subTest(split=split, family=family):
                    validate_family_label(family, split, 1)

    def test_known_family_cannot_be_mislabeled_unknown(self):
        for family in KNOWN:
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'known font mislabeled unknown'):
                validate_family_label(family, 'train', 0)

    def test_unknown_family_cannot_be_mislabeled_known(self):
        with self.assertRaisesRegex(ValueError, 'unknown source mislabeled as known'):
            validate_family_label('Kaiti SC', 'train', 1)

    def test_unknown_family_holdout_is_enforced(self):
        for source_split, families in UNKNOWN_SPLITS.items():
            for family in families:
                validate_family_label(family, source_split, 0)
                for other in UNKNOWN_SPLITS:
                    if other != source_split:
                        with self.subTest(family=family, other=other), self.assertRaisesRegex(ValueError, 'family leakage'):
                            validate_family_label(family, other, 0)

    def test_bool_and_string_labels_are_not_integers(self):
        for value in (False, True, '0', '1', 1.0, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'integer unknown=0 / known=1'):
                validate_family_label('Kaiti SC', 'train', value)

    def test_test_partition_fails_before_reading_any_files(self):
        with self.assertRaisesRegex(ValueError, 'test pixels require'):
            load_partition('/path/that/does/not/exist', 'test')

    def test_reversed_class_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'MANIFEST.json').write_text(json.dumps({'schema': SCHEMA, 'labels': ['known', 'unknown']}))
            with self.assertRaisesRegex(ValueError, 'class order differs'):
                load_partition(temp, 'train')

    def test_preprocess_change_and_constant_change_affect_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp)/'original.py'
            changed = Path(temp)/'changed.py'
            original.write_text('import math\nMAX_TILES = 8\nMAX_PIXELS = 4000000\ndef preprocess_region(image):\n return image\n')
            changed.write_text(original.read_text().replace('MAX_TILES = 8', 'MAX_TILES = 4'))
            self.assertNotEqual(preprocessing_contract(original), preprocessing_contract(changed))
            changed.write_text(original.read_text().replace('return image', 'return None'))
            self.assertNotEqual(preprocessing_contract(original), preprocessing_contract(changed))

    def test_unrelated_runtime_method_does_not_change_preprocess_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp)/'original.py'
            changed = Path(temp)/'changed.py'
            original.write_text('import math\nMAX_TILES = 8\ndef preprocess_region(image):\n return image\n')
            changed.write_text(original.read_text() + '\nclass DifferentServingAPI:\n def predict(self):\n  return 0\n')
            self.assertEqual(preprocessing_contract(original), preprocessing_contract(changed))


if __name__ == '__main__':
    unittest.main()
