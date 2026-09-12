"""Keep extra training fonts and pixels separate from fixed evaluation sources."""
import json
from pathlib import Path
import tempfile
import unittest

from training.capture.generate_rejection_supplement import KNOWN_LATIN, NEW_UNKNOWN
from training.capture.generate_unknown_scenes import UNKNOWN_SPLITS
from training.capture.prepare_rejection_supplement import SCHEMA, load_partition, validate_label


class RejectionSupplementTests(unittest.TestCase):
    def test_new_fonts_are_disjoint_from_every_original_unknown_partition(self):
        self.assertFalse(set(NEW_UNKNOWN) & {f for fs in UNKNOWN_SPLITS.values() for f in fs})

    def test_training_labels_match_known_and_new_unknown_sources(self):
        for family in KNOWN_LATIN:
            validate_label(family, 1)
        for family in NEW_UNKNOWN:
            validate_label(family, 0)

    def test_reversed_labels_fail(self):
        for family, label in [('SF Pro', 0), ('Noteworthy', 1)]:
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'family/label mismatch'):
                validate_label(family, label)

    def test_original_calibration_and_test_families_cannot_be_added(self):
        for family in UNKNOWN_SPLITS['calibration'] + UNKNOWN_SPLITS['test']:
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'family/label mismatch'):
                validate_label(family, 0)

    def test_non_training_split_is_rejected(self):
        for split in ['calibration', 'test']:
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'may not contain calibration or test'):
                validate_label('Noteworthy', 0, split)

    def test_bool_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer unknown=0/known=1'):
            validate_label('Noteworthy', False)

    def test_loader_cannot_accept_evaluation_partition_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'MANIFEST.json'
            p.write_text(json.dumps({'schema': SCHEMA, 'training_only': True, 'splits': {'train': {}, 'test': {}}}))
            with self.assertRaisesRegex(ValueError, 'training-only source'):
                load_partition(temp)


if __name__ == '__main__':
    unittest.main()
