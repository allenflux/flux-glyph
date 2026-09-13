"""Replay 16 unknown rows across all 11 TRAIN sources without fixed old/new shares."""
from collections import Counter, defaultdict

import numpy as np

from retention_supplement_sampler import (
    IOS_ANCHOR_FAMILIES, NEW_SOURCES, SUPPLEMENT_MARKER, VIEW_WEIGHTS,
)
from train_regions import require
from retention_source_balanced_sampler import SourceBalancedSampler, SAMPLING as SOURCE_SAMPLING, expected_unknown_source_counts

KNOWN_FAMILIES=('LXGW WenKai','WenQuanYi Micro Hei')
KNOWN_SUPPLEMENT_MARKER='_retention_known_supplement'


SAMPLING={**SOURCE_SAMPLING,'known':48,'base_known':46,'known_supplement':2,
    'known_supplement_families':list(KNOWN_FAMILIES),'known_supplement_rows_per_source_per_batch':1,
    'known_sampling':'For each paired class, one old TRAIN row plus one new paired known row; other 22 classes retain two old rows.',
    'known_supplement_selection':'Weighted available view then uniform native region; actual Regular source faces only.',
    'known_class_prior_changed':False,
    'limitation':'The two new sources are Regular only. Half of each paired class base slots now use Regular; other weights are not established by these new captures.'}


class PairedKnownSampler(SourceBalancedSampler):
    def __init__(self,oldrows,families,seed,train_pools,newrows,knownrows):
        require(not any(KNOWN_SUPPLEMENT_MARKER in row for row in [*oldrows,*newrows]),'Input already contains a known cache marker')
        super().__init__(oldrows,families,seed,train_pools,newrows)
        self.known_pools=defaultdict(list)
        for raw in knownrows:
            family=raw.get('family');pair=raw.get('pair_evidence',{})
            require(family in KNOWN_FAMILIES and raw.get('target')==families.index(family)
                and raw.get('source_font_family')==family and raw.get('split')=='train' and raw.get('domain')=='android'
                and raw.get('source_dataset')=='android_paired_known_supplement' and raw.get('native_font_verified') is True
                and raw.get('view') in VIEW_WEIGHTS and pair.get('intentional_train_text_pair') is True
                and raw.get('font_face') and len(raw.get('font_file_sha256',''))==64
                and type(raw.get('tile_start')) is int and raw['tile_start']>=0
                and type(raw.get('tile_count')) is int and 1<=raw['tile_count']<=8
                and KNOWN_SUPPLEMENT_MARKER not in raw and SUPPLEMENT_MARKER not in raw,
                'Paired known sampling requires verified named TRAIN images with exact cache mapping')
            self.known_pools[(family,raw['view'])].append(dict(raw,**{KNOWN_SUPPLEMENT_MARKER:True}))
        require(set(self.known_pools)=={(f,v) for f in KNOWN_FAMILIES for v in VIEW_WEIGHTS},'Each paired known source needs all views')
        self.known_counts=Counter();self.known_view_counts=Counter();self.known_position=0

    def batch(self):
        rows = []; visits=Counter()
        for _ in range(48):
            target = self.base.known_targets[self.known_position % len(self.base.known_targets)]
            self.known_position += 1
            family=self.base.families[target];visits[family]+=1
            if family in KNOWN_FAMILIES and visits[family]==2:
                view=str(self.rng.choice(self.supplement_views,p=self.supplement_weights))
                pool=self.known_pools[(family,view)]
                rows.append(pool[int(self.rng.integers(len(pool)))])
                self.known_counts[family]+=1;self.known_view_counts[view]+=1
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

        self.slot_counts.update({'base_known':46,'known_supplement':2,'base_unknown': 16 - supplement_count,
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
        return {**super().report(), 'schema':'flux-glyph-retention-paired-known-sampling-v1',
            'known_supplement_rows':sum(self.known_counts.values()),'known_supplement_source_rows':dict(self.known_counts),
            'known_supplement_view_rows':dict(self.known_view_counts),'known_class_prior_changed':False,
            'source_balanced': True, 'unknown_rows': self.unknown_position,
            'unknown_source_order': self.unknown_source_order,
            'unknown_source_rows': dict(self.unknown_source_counts)}
