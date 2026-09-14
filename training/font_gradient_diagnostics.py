"""Read-only decomposition and local gradient geometry for the frozen font loss."""
from __future__ import annotations

import numpy as np

from train_regions import require

TERMS = ('replay_font', 'focus_font', 'mobile_font', 'size', 'teacher',
         'angular', 'unknown_oe', 'pair')
REFERENCES = {'font_supervision': (1., 1., 1., 0., 0., 0., 0., 0.),
              'mobile_font': (0., 0., 1., 0., 0., 0., 0., 0.)}
GROUPS = ('trunk', 'style', 'shared', 'family_head', 'size_head', 'all')
MIN_NORM = 1e-12


def decompose(model, batch):
    """Each term already includes its actual weight; sum equals the frozen loss."""
    import torch
    import torch.nn.functional as F
    import train_unified_native_pairs as original

    font, size, features = original.forward_with_features(model, batch['images'])
    require(font.shape == (180, 25) and size.shape == (180,), 'Expected one 180-row font forward')
    replay = original.full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        batch['targets'], batch['sizes'], batch['teacher'], batch['rows'], original.FAMILIES)
    focus = original.focus_full_loss(font[96:128], size[96:128], features[96:128],
        model.family_head.weight, batch['focus_targets'], batch['focus_sizes'],
        batch['focus_teacher'], batch['focus_rows'])
    mobile_ce = F.cross_entropy(font[128:148], batch['mobile_targets'])
    mobile_size = F.smooth_l1_loss(size[128:148], batch['mobile_sizes'], beta=.05)
    oe = original.unknown_named_uniformity(font[:96], batch['targets'], batch['rows'], original.FAMILIES)
    pair = original.pair_margin_loss(font[148:180], batch['pair_targets'])
    require(original.FOCUS_WEIGHT == original.MOBILE_WEIGHT == .5
            and original.OE_MULTIPLIER == .25 and original.PAIR_WEIGHT == .1,
            'Frozen objective coefficients changed')
    terms = dict(zip(TERMS, (
        replay[1], .5 * focus[1], .5 * mobile_ce,
        .2 * replay[2] + .1 * focus[2] + .1 * mobile_size,
        2. * replay[3] + focus[3], replay[4] + .5 * focus[4], .25 * oe, .1 * pair)))
    original_total = replay[0] + .5 * focus[0] + .25 * oe + .5 * (mobile_ce + .2 * mobile_size) + .1 * pair
    require(all(value.ndim == 0 and bool(torch.isfinite(value)) for value in terms.values())
            and torch.allclose(sum(terms.values()), original_total, rtol=2e-6, atol=2e-6),
            'Weighted component sum differs from the frozen objective')
    return original_total, terms


def geometry(gram):
    """Gram matrices use weighted gradients, never scalar loss magnitudes."""
    g = np.asarray(gram, dtype=np.float64)
    n = len(TERMS)
    require(g.shape == (n, n) and np.isfinite(g).all()
            and np.allclose(g, g.T, atol=1e-12, rtol=1e-10)
            and np.diag(g).min() >= -1e-12, 'Invalid gradient Gram matrix')
    norms = np.sqrt(np.maximum(np.diag(g), 0.))
    cosines = {}
    for i, name in enumerate(TERMS):
        cosines[name] = {other: (float(np.clip(g[i, j] / (norms[i] * norms[j]), -1, 1))
            if norms[i] > MIN_NORM and norms[j] > MIN_NORM else None)
            for j, other in enumerate(TERMS)}
    references = {}
    for name, coefficients in REFERENCES.items():
        c = np.asarray(coefficients, dtype=np.float64)
        square = float(c @ g @ c); norm = float(np.sqrt(max(square, 0.)))
        dots = c @ g
        # For raw SGD, delta L_reference / learning_rate ~= -g_reference dot update_gradient.
        # AdamW momentum, normalization, curvature and validation behavior are not inferred here.
        references[name] = {'norm': norm,
            'term_cosine': {term: (float(np.clip(dots[i] / (norm * norms[i]), -1, 1))
                if norm > MIN_NORM and norms[i] > MIN_NORM else None) for i, term in enumerate(TERMS)},
            'term_norm_ratio': {term: float(norms[i] / norm) if norm > MIN_NORM else None
                                for i, term in enumerate(TERMS)},
            'term_dot_relative_to_reference_squared': {
                term: float(dots[i] / square) if norm > MIN_NORM else None for i, term in enumerate(TERMS)},
            'full_descent_alignment': float(dots.sum() / square) if norm > MIN_NORM else None,
            'without_size_descent_alignment': float((dots.sum() - dots[3]) / square) if norm > MIN_NORM else None,
            'without_teacher_descent_alignment': float((dots.sum() - dots[4]) / square) if norm > MIN_NORM else None}
    return {'weighted_norms': dict(zip(TERMS, map(float, norms))), 'cosines': cosines,
            'references': references}


def gradients(model, total, terms):
    """Return per-group geometry without modifying weights, buffers or .grad."""
    import torch
    require(tuple(terms) == TERMS, 'Gradient component order differs')
    named = list(model.named_parameters()); parameters = tuple(p for _, p in named)
    require(all(p.grad is None and p.requires_grad for p in parameters),
            'Diagnostic requires untouched, trainable parameter gradients')
    arrays = []
    for loss in terms.values():
        values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        arrays.append([np.zeros(p.numel(), dtype=np.float64) if value is None else
                       value.detach().cpu().numpy().astype(np.float64).reshape(-1)
                       for value, p in zip(values, parameters)])
    direct = torch.autograd.grad(total, parameters, allow_unused=True)
    error_sq = 0.; direct_sq = 0.; max_error = 0.
    for j, (value, parameter) in enumerate(zip(direct, parameters)):
        actual = np.zeros(parameter.numel()) if value is None else value.detach().cpu().numpy().astype(np.float64).reshape(-1)
        difference = sum(row[j] for row in arrays) - actual
        require(np.isfinite(actual).all() and np.isfinite(difference).all(), 'Nonfinite component gradient')
        error_sq += float(np.sum(difference * difference))
        direct_sq += float(np.sum(actual * actual))
        max_error = max(max_error, float(np.abs(difference).max()))
    relative_error = float(np.sqrt(error_sq) / max(np.sqrt(direct_sq), MIN_NORM))
    require(relative_error <= 5e-5 and max_error <= 2e-4,
            'Component gradients do not reconstruct the original total gradient')
    result = {}
    for group in GROUPS:
        chosen = [j for j, (name, _) in enumerate(named) if group == 'all'
            or (group == 'shared' and name.split('.')[0] in ('trunk', 'style'))
            or name.startswith(group + '.')]
        require(chosen, 'Missing model parameter group: ' + group)
        values = np.stack([np.concatenate([row[j] for j in chosen]) for row in arrays])
        # Explicit float64 contraction avoids a spurious large-matmul floating-point
        # warning in the local NumPy/Accelerate build, without hiding finite checks.
        gram = np.einsum('ik,jk->ij', values, values, optimize=False)
        result[group] = {**geometry(gram), 'gram': gram.tolist(), 'parameters': values.shape[1]}
    require(all(p.grad is None for p in parameters), 'Diagnostic populated optimizer gradients')
    return result, {'relative_l2_error': relative_error, 'maximum_absolute_error': max_error}


def summary(records):
    """Descriptive batch summaries; these are not independent validation samples."""
    require(records, 'No gradient measurements')
    def stats(values):
        valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
        return {'defined_batches': len(valid), 'undefined_batches': len(values) - len(valid),
            'mean': float(valid.mean()) if len(valid) else None,
            'median': float(np.median(valid)) if len(valid) else None,
            'minimum': float(valid.min()) if len(valid) else None,
            'maximum': float(valid.max()) if len(valid) else None,
            'negative_fraction': float((valid < 0).mean()) if len(valid) else None}
    result = {}
    for group in GROUPS:
        values = [row['groups'][group] for row in records]
        refs = {}
        for reference in REFERENCES:
            rows = [v['references'][reference] for v in values]
            refs[reference] = {key: {term: stats([r[key][term] for r in rows]) for term in TERMS}
                for key in ('term_cosine', 'term_norm_ratio', 'term_dot_relative_to_reference_squared')}
            refs[reference].update({key: stats([r[key] for r in rows]) for key in
                ('full_descent_alignment', 'without_size_descent_alignment', 'without_teacher_descent_alignment')})
        result[group] = {'references': refs,
            'weighted_norms': {term: stats([v['weighted_norms'][term] for v in values]) for term in TERMS}}
    return result
