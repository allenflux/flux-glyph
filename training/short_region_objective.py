"""Training-only short-region supervision and explicit R22 regression targets."""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from prepare_unified_regions import FAMILIES
from train_regions import require

SHORT_BATCH = 32
SHORT_WEIGHT = .5
SHORT_UNKNOWN_SOURCES = ('Apple Chancery', 'Bradley Hand', 'Courier', 'Lato',
                         'Liu Jian Mao Cao', 'Noteworthy', 'Open Sans', 'Smiley Sans', 'Times')
SHORT_SAMPLING = {
    'known_per_batch': 24, 'unknown_per_batch': 8,
    'known_selection': 'one row per named family; cycle available glyph counts, domains and faces',
    'unknown_selection': 'continuous cycle of existing TRAIN unknown source families',
    'views': {'native': .4, 'half': .2, 'three_quarters_jpeg75': .2, 'jpeg75': .2},
    'labels': 'verified parent TRAIN font identity, unchanged',
    'unknown_sources': list(SHORT_UNKNOWN_SOURCES),
}
EXTRA_ACCEPTANCE = {
    'short_known_coverage_gain': .02,
    'short_unknown_false_names_increase': 0,
    'all_known_coverage_drop': 0.,
    'all_named_precision_drop': 0.,
    'all_unknown_withholding_drop': 0.,
    'per_domain_known_coverage_drop': .005,
    'original_46_checks_required': True,
    'original_18_development_checks_required': True,
    'r22_18_development_checks_also_required': True,
    'blind_test': False,
}


class ShortSampler:
    def __init__(self, rows, seed):
        self.rng = np.random.default_rng(seed)
        self.pools = defaultdict(list)
        for row in rows:
            require(row.get('split') == 'train' and row.get('native_font_verified') is True
                    and type(row.get('target')) is int and FAMILIES[row['target']] == row.get('family')
                    and row.get('view') in SHORT_SAMPLING['views'], 'Short sampler requires verified TRAIN rows')
            count = row['glyph_count']
            require(type(count) is int and 1 <= count <= 4, 'Short glyph count differs')
            source = row['source_font_family'] if row['family'] == '__unknown__' else row['family']
            self.pools[(source, count, row['domain'], row['font_face'], row['view'])].append(row)
        self.sources = sorted({k[0] for k in self.pools})
        self.unknown_sources = sorted({r['source_font_family'] for r in rows if r['family'] == '__unknown__'})
        require(set(FAMILIES[:-1]) <= set(self.sources) and self.unknown_sources == list(SHORT_UNKNOWN_SOURCES)
                and not set(self.unknown_sources) & set(FAMILIES), 'Missing or relabeled short font sources')
        self.lengths = {s: sorted({k[1] for k in self.pools if k[0] == s}) for s in self.sources}
        self.faces = {(s, n): sorted({(k[2], k[3]) for k in self.pools if k[:2] == (s, n)})
                      for s in self.sources for n in self.lengths[s]}
        self.positions = Counter(); self.face_positions = Counter(); self.counts = Counter()
        self.unknown_position = 0; self.steps = 0

    def _row(self, source):
        lengths = self.lengths[source]
        count = lengths[self.positions[source] % len(lengths)]
        self.positions[source] += 1
        faces = self.faces[(source, count)]
        domain, face = faces[self.face_positions[(source, count)] % len(faces)]
        self.face_positions[(source, count)] += 1
        views = [v for v in SHORT_SAMPLING['views'] if (source, count, domain, face, v) in self.pools]
        weights = np.array([SHORT_SAMPLING['views'][v] for v in views])
        view = str(self.rng.choice(views, p=weights / weights.sum()))
        pool = self.pools[(source, count, domain, face, view)]
        row = pool[int(self.rng.integers(len(pool)))]
        self.counts[(row['family'], source, count, domain, face, view)] += 1
        return row

    def batch(self):
        result = [self._row(family) for family in FAMILIES[:-1]]
        for _ in range(8):
            source = self.unknown_sources[self.unknown_position % len(self.unknown_sources)]
            self.unknown_position += 1
            result.append(self._row(source))
        self.rng.shuffle(result); self.steps += 1
        return result

    def report(self):
        return {'steps': self.steps, 'rows': sum(self.counts.values()),
                'unknown_source_order': self.unknown_sources,
                'counts': [dict(family=f, source_font_family=s, glyph_count=n, domain=d, font_face=face, view=v, rows=c)
                           for (f, s, n, d, face, v), c in sorted(self.counts.items())]}


def short_loss(logits, ratios, targets, sizes, rows):
    """Only actual short-crop truth is supervised; no teacher from its parent tile."""
    import torch
    from torch.nn import functional as F
    from retention_confidence_floor_loss import unknown_floor_loss
    require(logits.shape == (32, 25) and ratios.shape == targets.shape == sizes.shape == (32,)
            and len(rows) == 32 and targets.dtype == torch.int64
            and all(bool(torch.isfinite(v).all()) for v in (logits, ratios, sizes)), 'Invalid short supervision tensors')
    for row, target in zip(rows, targets.detach().cpu().tolist()):
        require(row['split'] == 'train' and row['native_font_verified'] is True
                and row['target'] == target and row['family'] == FAMILIES[target]
                and 1 <= row['glyph_count'] <= 4, 'Short supervision cannot relabel a crop')
    known = targets != 24
    ce = (F.cross_entropy(logits[known], targets[known], label_smoothing=.03, reduction='sum')
          if bool(known.any()) else logits.sum() * 0.)
    unknown = (unknown_floor_loss(logits[~known], 24).sum()
               if bool((~known).any()) else logits.sum() * 0.)
    font = (ce + unknown) / SHORT_BATCH
    size = F.smooth_l1_loss(ratios, sizes, beta=.05)
    return SHORT_WEIGHT * (font + .2 * size), font, size


def short_population(details, rows):
    require(len(details) == len(rows), 'CAL rows and decisions differ')
    pairs = [(d, r) for d, r in zip(details, rows) if r['tile_count'] == 1]
    known = [d for d, r in pairs if r['target'] != 24]
    unknown = [d for d, r in pairs if r['target'] == 24]
    require(known and unknown, 'CAL lacks the fixed short-region populations')
    return {'known_views': len(known), 'correct_named': sum(d['correct_named'] for d in known),
            'known_correct_coverage': sum(d['correct_named'] for d in known) / len(known),
            'unknown_views': len(unknown), 'unknown_falsely_named': sum(d['named'] for d in unknown)}


def compare_r22(current, baseline, current_short, baseline_short):
    require(all(current_short[k] == baseline_short[k] for k in ('known_views', 'unknown_views')),
            'Short-region comparison denominators changed')
    checks = []
    def minimum(name, actual, bound):
        checks.append({'name': name, 'actual': actual, 'minimum': bound,
                       'passed': actual is not None and actual + 1e-12 >= bound})
    minimum('short_known_coverage', current_short['known_correct_coverage'],
            baseline_short['known_correct_coverage'] + EXTRA_ACCEPTANCE['short_known_coverage_gain'])
    checks.append({'name': 'short_unknown_false_names', 'actual': current_short['unknown_falsely_named'],
                   'maximum': baseline_short['unknown_falsely_named'],
                   'passed': current_short['unknown_falsely_named'] <= baseline_short['unknown_falsely_named']})
    for key in ('known_correct_coverage', 'named_precision', 'unknown_not_named_rate'):
        minimum('all.' + key, current[key], baseline[key])
    for domain in ('ios', 'android'):
        minimum(domain + '.known_correct_coverage', current['per_domain'][domain]['known_correct_coverage'],
                baseline['per_domain'][domain]['known_correct_coverage'] - .005)
    return {'passed': all(row['passed'] for row in checks), 'checks': checks, 'policy': EXTRA_ACCEPTANCE}
