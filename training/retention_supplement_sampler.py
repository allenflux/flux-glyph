"""Balance verified new TRAIN negatives while retaining the original known replay."""
from collections import Counter, defaultdict
import numpy as np

from train_unified_retention import RetentionSampler, IOS_ANCHOR_FAMILIES
from train_unified_regions import VIEW_WEIGHTS
from train_regions import require

NEW_SOURCES = ('WenQuanYi Zen Hei', 'Zhuque Fangsong')
SUPPLEMENT_MARKER = '_retention_supplement'
SAMPLING = {
    'batch_size': 96, 'base_known': 48, 'base_unknown': 8, 'supplement_unknown': 8,
    'ios_native_pingfang': 16, 'ios_native_sfpro_helvetica': 8,
    'ios_native_original_eight': 8, 'supplement_sources': list(NEW_SOURCES),
    'supplement_rows_per_source_per_batch': 4,
    'supplement_view_weights': dict(VIEW_WEIGHTS),
    'supplement_selection': 'Cycle both new source families; weighted view then uniform verified TRAIN region.',
    'original_eight_families': IOS_ANCHOR_FAMILIES,
    'platform_labels_are_network_inputs': False,
    'replacement_proposals_counted_as_training': False,
}


class SupplementSampler(RetentionSampler):
    def __init__(self, rows, families, seed, train_pools, supplement_rows):
        require(not any(SUPPLEMENT_MARKER in row for row in rows), 'Original rows contain a cache marker')
        super().__init__(rows, families, seed, train_pools)
        require(not set(NEW_SOURCES) & set(self.base.unknown_families), 'New source is already an original TRAIN negative')
        self.supplement_pools = defaultdict(list)
        for source in supplement_rows:
            require(source.get('split') == 'train' and source.get('domain') == 'android'
                    and source.get('family') == '__unknown__' and source.get('target') == families.index('__unknown__')
                    and source.get('native_font_verified') is True and source.get('source_font_family') in NEW_SOURCES
                    and source.get('view') in VIEW_WEIGHTS and SUPPLEMENT_MARKER not in source
                    and type(source.get('tile_start')) is int and source['tile_start'] >= 0
                    and type(source.get('tile_count')) is int and 1 <= source['tile_count'] <= 8
                    and isinstance(source.get('font_face'), str) and source['font_face']
                    and isinstance(source.get('font_file_sha256'), str) and len(source['font_file_sha256']) == 64,
                    'Supplement sampling requires verified, mapped TRAIN unknown regions')
            row = dict(source, **{SUPPLEMENT_MARKER: True})
            self.supplement_pools[(row['source_font_family'], row['view'])].append(row)
        require(set(self.supplement_pools) == {(family, view) for family in NEW_SOURCES for view in VIEW_WEIGHTS},
                'Both new sources must have each declared native/derived view')
        self.supplement_counts = Counter()
        self.source_counts = Counter()
        self.supplement_view_counts = Counter()

    def batch(self):
        rows = []
        for _ in range(48):
            target = self.base.known_targets[self.base.known_position % len(self.base.known_targets)]
            self.base.known_position += 1
            rows.append(self.base._row(('known', target)))
        for _ in range(8):
            source = self.base.unknown_families[self.base.unknown_position % len(self.base.unknown_families)]
            self.base.unknown_position += 1
            rows.append(self.base._row(('unknown', source)))
        views = sorted(VIEW_WEIGHTS)
        weights = np.asarray([VIEW_WEIGHTS[view] for view in views], dtype=np.float64)
        weights /= weights.sum()
        for family in NEW_SOURCES:
            for _ in range(4):
                view = str(self.rng.choice(views, p=weights))
                pool = self.supplement_pools[(family, view)]
                rows.append(pool[int(self.rng.integers(len(pool)))])
                self.supplement_counts[family] += 1
                self.supplement_view_counts[view] += 1
        rows.extend(self._focus_row('PingFang', 'ios_native_pingfang') for _ in range(16))
        for _ in range(8):
            family = ['SF Pro', 'Helvetica'][self.extra_positions['system'] % 2]
            self.extra_positions['system'] += 1
            rows.append(self._focus_row(family, 'ios_native_latin'))
        for _ in range(8):
            family = IOS_ANCHOR_FAMILIES[self.extra_positions['original_eight'] % 8]
            self.extra_positions['original_eight'] += 1
            rows.append(self._focus_row(family, 'ios_native_anchor_original8'))
        self.slot_counts.update({key: SAMPLING[key] for key in ('base_known', 'base_unknown', 'supplement_unknown',
            'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')})
        for row in rows:
            self.family_counts[row['family']] += 1
            self.domain_counts[row['domain']] += 1
            self.view_counts[row['view']] += 1
            self.source_counts[(row['domain'], row['source_font_family'], row['family'])] += 1
            self.face_counts[(row['family'], row['domain'], row['font_face'],
                str(row.get('font_file_sha256') or ''), row.get('ttc_index', 0))] += 1
        require(len(rows) == 96, 'Incomplete supplemental replay batch')
        self.rng.shuffle(rows)
        return rows

    def report(self):
        return {**super().report(), 'schema': 'flux-glyph-retention-supplement-sampling-v1',
            'supplement_source_rows': dict(self.supplement_counts),
            'supplement_view_rows': dict(self.supplement_view_counts),
            'source_rows': [{'domain': d, 'source_font_family': f, 'target_family': t, 'rows': n}
                for (d, f, t), n in sorted(self.source_counts.items())],
            'test_read': False, 'development_holdout_read': False, 'proposal_rows_discarded': 0}
