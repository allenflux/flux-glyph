"""Balance paired WenKai TRAIN rows across three verified font faces."""
from collections import Counter, defaultdict

from retention_paired_known_sampler import (
    IOS_ANCHOR_FAMILIES, KNOWN_FAMILIES, KNOWN_SUPPLEMENT_MARKER,
    NEW_SOURCES, PairedKnownSampler, SAMPLING as PAIRED_SAMPLING,
    SUPPLEMENT_MARKER, VIEW_WEIGHTS,
)
from train_regions import require

WENKAI = 'LXGW WenKai'
MICRO = 'WenQuanYi Micro Hei'
FACE_WEIGHTS = {WENKAI: ('Light', 'Medium', 'Regular'), MICRO: ('Regular',)}
FACE_NAMES = {WENKAI: {'Light': 'LXGWWenKai-Light', 'Medium': 'LXGWWenKai-Medium',
                       'Regular': 'LXGWWenKai-Regular'},
              MICRO: {'Regular': 'WenQuanYiMicroHei'}}

SAMPLING = {**PAIRED_SAMPLING,
    'known_supplement_selection':
        'WenKai cycles Light, Medium, Regular exactly; Micro uses its sole Regular face; then weighted view and uniform region.',
    'known_supplement_face_cycle': {family: list(weights) for family, weights in FACE_WEIGHTS.items()},
    'known_supplement_wenkai_rows_per_face_at_6000_steps': 2000,
    'known_total_wenkai_rows_per_face_at_6000_steps': 4000,
    'limitation': 'Paired Micro has one verified face; paired WenKai has exactly Light, Medium and Regular.'}


def _weight(family, face):
    matches = [weight for weight, name in FACE_NAMES.get(family, {}).items() if face == name]
    require(len(matches) == 1, 'Paired known font_face must match the exact verified family face identity')
    return matches[0]


class FaceBalancedSampler(PairedKnownSampler):
    """Keep the 96-slot replay while cycling paired WenKai faces exactly."""

    def __init__(self, oldrows, families, seed, train_pools, newrows, knownrows):
        super().__init__(oldrows, families, seed, train_pools, newrows, knownrows)
        expected_offset = 0
        identities = defaultdict(dict)
        face_pools = defaultdict(list)
        row_keys = set()
        for raw in knownrows:
            family = raw['family']; face = raw['font_face']; weight = _weight(family, face)
            identity = (face, raw['font_file_sha256'], raw.get('ttc_index', 0))
            pair = raw['pair_evidence']
            key = (raw.get('source_id'), raw.get('region_id'), raw['view'])
            require(raw['tile_start'] == expected_offset and key not in row_keys
                    and isinstance(key[0], str) and key[0] and isinstance(key[1], str) and key[1]
                    and pair.get('intentional_train_text_pair') is True
                    and pair.get('training_family') == '__unknown__'
                    and isinstance(pair.get('source_id'), str) and pair['source_id']
                    and isinstance(pair.get('region_id'), str) and pair['region_id']
                    and isinstance(pair.get('font_family'), str) and pair['font_family']
                    and type(identity[2]) is int and identity[2] >= 0,
                    'Merged paired rows must retain unique identities and contiguous reindexed tile offsets')
            expected_offset += raw['tile_count']; row_keys.add(key)
            require(weight in FACE_WEIGHTS[family], 'Paired known family has an undeclared font face')
            if weight in identities[family]:
                require(identities[family][weight] == identity,
                        'One paired face weight maps to multiple font identities')
            identities[family][weight] = identity
            face_pools[(family, weight, raw['view'])].append(
                dict(raw, **{KNOWN_SUPPLEMENT_MARKER: True}))
        require(set(identities) == set(KNOWN_FAMILIES)
                and all(set(identities[family]) == set(FACE_WEIGHTS[family]) for family in KNOWN_FAMILIES),
                'Paired rows must contain exactly three WenKai faces and one Micro face')
        require(set(face_pools) == {(family, weight, view) for family in KNOWN_FAMILIES
                for weight in FACE_WEIGHTS[family] for view in VIEW_WEIGHTS},
                'Every paired face must contain every weighted TRAIN view')
        self.known_face_pools = face_pools
        self.known_face_identities = identities
        self.known_face_positions = Counter(dict.fromkeys(KNOWN_FAMILIES, 0))
        self.known_face_counts = Counter()
        self.known_face_view_counts = Counter()
        self.known_tile_count = expected_offset

    def _paired_row(self, family):
        position = self.known_face_positions[family]
        weight = FACE_WEIGHTS[family][position % len(FACE_WEIGHTS[family])]
        self.known_face_positions[family] += 1
        view = str(self.rng.choice(self.supplement_views, p=self.supplement_weights))
        pool = self.known_face_pools[(family, weight, view)]
        row = pool[int(self.rng.integers(len(pool)))]
        identity = self.known_face_identities[family][weight]
        require((row['font_face'], row['font_file_sha256'], row.get('ttc_index', 0)) == identity,
                'Paired known row changed after face routing')
        self.known_counts[family] += 1
        self.known_view_counts[view] += 1
        self.known_face_counts[(family, *identity)] += 1
        self.known_face_view_counts[(family, *identity, view)] += 1
        return row

    def batch(self):
        rows = []; visits = Counter()
        for _ in range(48):
            target = self.base.known_targets[self.known_position % len(self.base.known_targets)]
            self.known_position += 1
            family = self.base.families[target]; visits[family] += 1
            if family in KNOWN_FAMILIES and visits[family] == 2:
                rows.append(self._paired_row(family))
            else:
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

        self.slot_counts.update({'base_known': 46, 'known_supplement': 2,
            'base_unknown': 16 - supplement_count, 'supplement_unknown': supplement_count,
            'ios_native_pingfang': 16, 'ios_native_sfpro_helvetica': 8,
            'ios_native_original_eight': 8})
        for row in rows:
            self.family_counts[row['family']] += 1
            self.domain_counts[row['domain']] += 1
            self.view_counts[row['view']] += 1
            self.source_counts[(row['domain'], row['source_font_family'], row['family'])] += 1
            self.face_counts[(row['family'], row['domain'], row['font_face'],
                str(row.get('font_file_sha256') or ''), row.get('ttc_index', 0))] += 1
        require(len(rows) == 96, 'Incomplete face-balanced replay batch')
        self.rng.shuffle(rows)
        return rows

    def report(self):
        report = super().report()
        report.update(schema='flux-glyph-retention-face-balanced-sampling-v1',
            known_supplement_tile_count=self.known_tile_count,
            known_supplement_face_rows=[{'family': family, 'font_face': face,
                'font_file_sha256': digest, 'ttc_index': ttc, 'rows': count}
                for (family, face, digest, ttc), count in sorted(self.known_face_counts.items())],
            known_supplement_face_view_rows=[{'family': family, 'font_face': face,
                'font_file_sha256': digest, 'ttc_index': ttc, 'view': view, 'rows': count}
                for (family, face, digest, ttc, view), count in sorted(self.known_face_view_counts.items())])
        return report
