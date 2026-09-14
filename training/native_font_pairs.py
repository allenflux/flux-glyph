"""Verified same-text font pairs for training a single font classifier."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from native_mobile_short_training import KNOWN, UNKNOWN_SOURCES, VIEWS
from prepare_unified_regions import FAMILIES
from train_regions import require, sha

SOURCES = (*KNOWN, *UNKNOWN_SOURCES)
POLICY = {'schema': 'flux-glyph-native-font-pair-margin-v1', 'pairs_per_batch': 16,
    'rows_per_batch': 32, 'weight': .1, 'margin': 1.,
    'loss': 'mean softplus(1 - ((L_a[y_a]-L_a[y_b])-(L_b[y_a]-L_b[y_b])))',
    'targets': 'different verified canonical font targets; never two unknown targets',
    'class_bias_cancels': True, 'extra_cross_entropy': False, 'extra_size_loss': False,
    'extra_parameters': False, 'runtime_changes': False, 'model_count': 1}
SAMPLING = {'schema': 'flux-glyph-native-font-pair-sampling-v1', 'pairs_per_batch': 16,
    'sources': list(SOURCES), 'anchor_source': 'one global round robin over all verified sources',
    'slot': 'script/length round robin within anchor source',
    'counterpart': 'round robin over different-target sources with an exact shared cohort',
    'cohort': 'round robin within anchor slot and counterpart source',
    'face': 'round robin within cohort and source', 'native_identity': 'uniform within chosen face',
    'view': 'same related view for both sides', 'view_weights': dict(zip(VIEWS, (.4, .2, .2, .2))),
    'rng_independent': True, 'platform_features': False, 'predictions_used_as_labels': False,
    'old_mobile_ce_pool_changed': False, 'related_views_are_independent_samples': False}


def source_family(row):
    return row['source_font_family'] if row['target'] == 24 else row['family']


def cohort_key(row):
    color = row['text_color_hex'].upper()
    require(color.startswith('#') and len(color) in (7, 9)
            and (len(color) == 7 or color.endswith('FF')), 'Pair text color must be opaque')
    int(color[1:], 16)
    px = float(row['native_font_size_px'])
    require(math.isfinite(px) and px > 0 and row['script'] in ('han', 'latin', 'numeric'),
            'Pair native style differs')
    # Exact native text additionally prevents case-folded hashes from pairing A with a.
    exact_text = hashlib.sha256(row['text'].encode()).hexdigest()
    return (row['normalized_text_sha256'], exact_text,
            'han' if row['script'] == 'han' else 'latin', row['glyph_count'], px, color[:7])


def build_pairs(mobile):
    cohorts = {}; originals = {}; targets = {}
    identities = set(); identity_proof = []
    for (_, script, length), faces in mobile['pools'].items():
        for groups in faces.values():
            for identity, views in groups.items():
                require(identity not in identities and set(views) == set(VIEWS),
                        'Pair native identity is duplicated or lacks four views')
                identities.add(identity)
                native = views['native']; source = source_family(native)
                require(native['split'] == 'train' and native['native_font_verified'] is True
                        and native['tile_count'] == 1 and source in SOURCES
                        and native['target'] == FAMILIES.index(native['family'])
                        and native['short_dataset'] in ('ios', 'android'), 'Pair native truth differs')
                key = cohort_key(native)
                require(key[2:4] == (script, length), 'Pair pool script/length differs')
                require(all(cohort_key(row) == key and source_family(row) == source
                            and row['target'] == native['target'] and row['tile_count'] == 1
                            and row['split'] == 'train' and row['native_font_verified'] is True
                            for row in views.values()), 'Pair related views disagree on native truth')
                require(source not in targets or targets[source] == native['target'],
                        'One native font source has conflicting targets')
                targets[source] = native['target']
                cohorts.setdefault(key, {}).setdefault(source, {}).setdefault(native['font_face'], {})[identity] = views
                originals.setdefault((source, script, length), set()).add(identity)
                identity_proof.append((list(identity), source, native['target'], list(key),
                    {v: views[v]['tiles_sha256'] for v in VIEWS}))
    require(set(targets) == set(SOURCES), 'Pair source population differs')
    eligible = {}; matched = set()
    for key, sources in cohorts.items():
        for source in sources:
            for other in sources:
                if targets[source] == targets[other]:
                    continue
                eligible.setdefault((source, key[2], key[3]), {}).setdefault(other, []).append(key)
                matched.update(identity for groups in sources[source].values() for identity in groups)
    require(set(eligible) == set(originals), 'A source/script/length has no different-target counterpart')
    for peers in eligible.values():
        for other in peers:
            peers[other].sort()
    proof = {'schema': 'flux-glyph-native-font-pair-pools-v1', 'sampling': SAMPLING, 'policy': POLICY,
        'native_identities': len(identities), 'pair_eligible_native_identities': len(matched),
        'unpaired_native_identities': len(identities - matched),
        'unpaired_remain_in_original_mobile_ce': True, 'cohorts': len(cohorts),
        'cohorts_with_different_targets': sum(len({targets[s] for s in values}) > 1 for values in cohorts.values()),
        'source_slot_count': len(eligible),
        'eligible_cohorts_by_slot_counterpart': {json.dumps((*slot, other)): len(keys)
            for slot, peers in sorted(eligible.items()) for other, keys in sorted(peers.items())},
        'native_identity_tile_proof_sha256': hashlib.sha256(json.dumps(sorted(identity_proof),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'bindings': {str(Path(__file__).resolve()): sha(__file__)},
        'model_inference': False, 'calibration_read': False, 'development_read': False, 'test_read': False}
    return {'cohorts': cohorts, 'eligible': eligible, 'targets': targets}, proof


class NativeFontPairSampler:
    def __init__(self, pools, seed):
        self.pools, self.rng = pools, np.random.default_rng(seed)
        self.steps = 0; self.cursor = 0
        self.slot_cursor = Counter(); self.peer_cursor = Counter(); self.cohort_cursor = Counter()
        self.face_cursor = Counter(); self.anchors = Counter(); self.counterparts = Counter()
        self.families = Counter(); self.sources = Counter(); self.views = Counter()
        self.slots = Counter(); self.pairs = Counter(); self.identities = Counter(); self.faces = Counter()

    def _identity(self, source, key):
        faces = self.pools['cohorts'][key][source]; names = sorted(faces)
        cursor = (source, key)
        face = names[self.face_cursor[cursor] % len(names)]; self.face_cursor[cursor] += 1
        groups = faces[face]; identities = sorted(groups)
        identity = identities[int(self.rng.integers(len(identities)))]
        self.identities[identity] += 1; self.faces[source, face] += 1
        return groups[identity]

    def _pair(self):
        source = SOURCES[self.cursor % len(SOURCES)]; self.cursor += 1
        slots = sorted(slot for slot in self.pools['eligible'] if slot[0] == source)
        slot = slots[self.slot_cursor[source] % len(slots)]; self.slot_cursor[source] += 1
        peers = self.pools['eligible'][slot]; names = sorted(peers)
        other = names[self.peer_cursor[slot] % len(names)]; self.peer_cursor[slot] += 1
        keys = peers[other]; cursor = (*slot, other)
        key = keys[self.cohort_cursor[cursor] % len(keys)]; self.cohort_cursor[cursor] += 1
        first, second = self._identity(source, key), self._identity(other, key)
        view = str(self.rng.choice(VIEWS, p=(.4, .2, .2, .2)))
        a, b = copy.deepcopy(first[view]), copy.deepcopy(second[view])
        require(a['target'] != b['target'] and cohort_key(a) == cohort_key(b)
                and a['view'] == b['view'] == view, 'Invalid matched font pair')
        self.anchors[source] += 1; self.counterparts[other] += 1; self.views[view] += 1
        self.slots[slot] += 1; self.pairs[source, other] += 1
        for row in (a, b):
            self.sources[source_family(row)] += 1; self.families[row['family']] += 1
        return a, b

    def batch(self):
        result = [self._pair() for _ in range(POLICY['pairs_per_batch'])]
        self.steps += 1
        return result

    def report(self):
        keyed = lambda values: {json.dumps(k): v for k, v in sorted(values.items())}
        return {'schema': SAMPLING['schema'], 'sampling': SAMPLING, 'steps': self.steps,
            'pairs': self.steps * 16, 'rows': self.steps * 32,
            'anchor_by_source': dict(self.anchors), 'counterpart_by_source': dict(self.counterparts),
            'rows_by_source': dict(self.sources), 'rows_by_family': dict(self.families),
            'pairs_by_view': dict(self.views), 'anchor_by_script_length': keyed(self.slots),
            'pairs_by_source': keyed(self.pairs), 'rows_by_identity': keyed(self.identities),
            'rows_by_face': keyed(self.faces)}


def pair_margin_loss(logits, targets):
    import torch
    import torch.nn.functional as F
    require(logits.shape == (32, 25) and targets.shape == (32,)
            and targets.dtype == torch.long and bool(torch.isfinite(logits).all())
            and bool(((targets >= 0) & (targets < 25)).all()), 'Expected32 aligned canonical pair rows')
    a, b = logits[::2], logits[1::2]; ya, yb = targets[::2], targets[1::2]
    require(bool((ya != yb).all()), 'Paired font targets must differ; two unknowns are not a negative pair')
    index = torch.arange(16, device=logits.device)
    difference = (a[index, ya] - a[index, yb]) - (b[index, ya] - b[index, yb])
    return F.softplus(POLICY['margin'] - difference).mean()
