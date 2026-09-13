"""Replay 16 unknown rows across all 11 TRAIN sources without fixed old/new shares."""
from collections import Counter

import numpy as np

from retention_supplement_sampler import (
    IOS_ANCHOR_FAMILIES, NEW_SOURCES, SupplementSampler, VIEW_WEIGHTS,
)
from train_regions import require


SAMPLING = {
    'batch_size': 96, 'base_known': 48, 'unknown': 16,
    'base_unknown': None, 'supplement_unknown': None,
    'source_balanced': True, 'unknown_source_count': 11,
    'old_unknown_source_count': 9, 'supplement_sources': list(NEW_SOURCES),
    'unknown_selection': 'Persistent round robin over sorted 11 TRAIN source families; no reset between batches.',
    'old_new_allocation': 'Dynamic actual counts from the combined source cycle, not fixed per-batch quotas.',
    'original_unknown_selection': 'Preserve source/domain/face cycling, weighted available view, then uniform region.',
    'supplement_selection': 'Preserve weighted view then uniform verified TRAIN region within the selected new source.',
    'supplement_view_weights': dict(VIEW_WEIGHTS),
    'ios_native_pingfang': 16, 'ios_native_sfpro_helvetica': 8,
    'ios_native_original_eight': 8, 'original_eight_families': list(IOS_ANCHOR_FAMILIES),
    'platform_labels_are_network_inputs': False,
    'replacement_proposals_counted_as_training': False,
}


def expected_unknown_source_counts(source_order, steps):
    """Exact cumulative row counts after whole 96-row batches, starting at zero."""
    require(type(steps) is int and steps >= 0, 'Steps must be a nonnegative integer')
    require(isinstance(source_order, (list, tuple)) and len(source_order) == 11
            and all(isinstance(source, str) and source for source in source_order),
            'Expected 11 named TRAIN unknown sources')
    require(list(source_order) == sorted(set(source_order))
            and set(NEW_SOURCES).issubset(source_order),
            'Unknown source order must contain the unique sorted old nine and new two sources')
    quotient, remainder = divmod(steps * SAMPLING['unknown'], len(source_order))
    return {source: quotient + int(index < remainder) for index, source in enumerate(source_order)}


class SourceBalancedSampler(SupplementSampler):
    def __init__(self, oldrows, families, seed, train_pools, newrows):
        super().__init__(oldrows, families, seed, train_pools, newrows)
        require(len(self.base.unknown_families) == 9,
                'Source-balanced replay requires exactly nine original TRAIN unknown families')
        self.unknown_sources = sorted([*self.base.unknown_families, *NEW_SOURCES])
        expected_unknown_source_counts(self.unknown_sources, 0)
        self.unknown_position = 0
        self.unknown_source_counts = Counter(dict.fromkeys(self.unknown_sources, 0))
        self.supplement_views = sorted(VIEW_WEIGHTS)
        self.supplement_weights = np.asarray(
            [VIEW_WEIGHTS[view] for view in self.supplement_views], dtype=np.float64)
        self.supplement_weights /= self.supplement_weights.sum()

    @property
    def unknown_source_order(self):
        return list(self.unknown_sources)

    def batch(self):
        rows = []
        for _ in range(48):
            target = self.base.known_targets[self.base.known_position % len(self.base.known_targets)]
            self.base.known_position += 1
            rows.append(self.base._row(('known', target)))

        supplement_count = 0
        for _ in range(16):
            source = self.unknown_sources[self.unknown_position % len(self.unknown_sources)]
            self.unknown_position += 1
            if source in NEW_SOURCES:
                view = str(self.rng.choice(self.supplement_views, p=self.supplement_weights))
                pool = self.supplement_pools[(source, view)]
                rows.append(pool[int(self.rng.integers(len(pool)))])
                self.supplement_counts[source] += 1
                self.supplement_view_counts[view] += 1
                supplement_count += 1
            else:
                # Keep the inherited base report restricted to rows actually drawn there.
                self.base.unknown_position += 1
                rows.append(self.base._row(('unknown', source)))
            self.unknown_source_counts[source] += 1

        rows.extend(self._focus_row('PingFang', 'ios_native_pingfang') for _ in range(16))
        for _ in range(8):
            family = ['SF Pro', 'Helvetica'][self.extra_positions['system'] % 2]
            self.extra_positions['system'] += 1
            rows.append(self._focus_row(family, 'ios_native_latin'))
        for _ in range(8):
            family = IOS_ANCHOR_FAMILIES[self.extra_positions['original_eight'] % 8]
            self.extra_positions['original_eight'] += 1
            rows.append(self._focus_row(family, 'ios_native_anchor_original8'))

        self.slot_counts.update({'base_known': 48, 'base_unknown': 16 - supplement_count,
            'supplement_unknown': supplement_count, 'ios_native_pingfang': 16,
            'ios_native_sfpro_helvetica': 8, 'ios_native_original_eight': 8})
        for row in rows:
            self.family_counts[row['family']] += 1
            self.domain_counts[row['domain']] += 1
            self.view_counts[row['view']] += 1
            self.source_counts[(row['domain'], row['source_font_family'], row['family'])] += 1
            self.face_counts[(row['family'], row['domain'], row['font_face'],
                str(row.get('font_file_sha256') or ''), row.get('ttc_index', 0))] += 1
        require(len(rows) == 96, 'Incomplete source-balanced replay batch')
        self.rng.shuffle(rows)
        return rows

    def report(self):
        return {**super().report(), 'schema': 'flux-glyph-retention-source-balanced-sampling-v1',
            'source_balanced': True, 'unknown_rows': self.unknown_position,
            'unknown_source_order': self.unknown_source_order,
            'unknown_source_rows': dict(self.unknown_source_counts)}
