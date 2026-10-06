"""Compile accepted source equations into runnable models, without user forms.

Prepared means structural/quote/data checks passed, not scientifically proven.
Rejected drafts and their reasons remain available for optional inspection.
"""
import copy
import numpy as np
from pydantic import ValidationError

from . import screening
from .evidence import validate_models
from .marginalization import eligibility
from .schema import Target


def prepare(bundle, context, query=None, allow_synthetic=False, require_query=False):
    target = Target.model_validate(bundle['target'])
    query = context if query is None else query
    accepted, skipped, seen = [], [], set()
    for index, raw in enumerate(bundle['models']):
        model = copy.deepcopy(raw)
        name = model.get('id') or f'model-{index + 1}'
        try:
            if name in seen:
                raise ValueError('Duplicate model identifier')
            seen.add(name)
            source_index = {s['id']: s for s in bundle['sources']}
            for anchor in [*(model.get('anchors') or []), *(a for d in model.get('missing_distributions') or [] for a in d.get('anchors') or [])]:
                source = source_index.get(anchor.get('source_id'), {})
                if not (allow_synthetic and source.get('kind') == 'synthetic') and source.get('dataset_review', {}).get('decision') not in ('approve', 'independent', 'background'):
                    raise ValueError('Source has not been approved')
            screening.check_models({**bundle, 'models': [model]})
            # The app accepts the model computationally on behalf of the user
            # who approved its sources. It does not forge a human model review.
            model['reviewed'] = True
            parsed = validate_models([model], bundle['sources'], target, context,
                                     allow_synthetic, check_data=False)[0]
            c_ok, _ = eligibility(context, parsed)
            q_ok = eligibility(query, parsed)[0] if require_query else np.ones(len(query), dtype=bool)
            if not c_ok.any() or not q_ok.any():
                _, reasons = eligibility(context if not c_ok.any() else query, parsed)
                details = list(dict.fromkeys(r for row in reasons for r in row))
                raise ValueError('; '.join(details) or 'No eligible rows')
            accepted.append(parsed.model_dump())
        except (ValueError, TypeError, KeyError) as error:
            if isinstance(error, ValidationError):
                reason = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in error.errors())
            else:
                reason = str(error)
            skipped.append({'model_id': name, 'reason': reason, 'draft': raw})
    coverage = {}
    for label, frame in [('context', context), *([('query', query)] if require_query else [])]:
        masks = [eligibility(frame, validate_models([m], bundle['sources'], target, frame,
                 allow_synthetic, check_data=False)[0])[0] for m in accepted]
        supported = np.any(masks, axis=0) if masks else np.zeros(len(frame), dtype=bool)
        coverage[label] = {'supported': int(supported.sum()), 'total': len(frame),
                           'skipped_positions': np.flatnonzero(~supported).tolist()}
    previous = bundle.get('preparation', {})
    return accepted, {'accepted': len(accepted), 'skipped': [*previous.get('skipped', []), *skipped], 'coverage': coverage,
                     'source_issues': previous.get('source_issues', []),
                     'query_handling': 'Literature coverage is not required for TabPFN test inputs.',
                     'approval_basis': 'Sources/models selected by the user or delegated workflow; software checked equations and mappings. No human model review claimed.'}
