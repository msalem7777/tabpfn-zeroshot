"""One-click extraction from approved papers with source-level failure isolation."""
import copy
import hashlib

from . import research, screening
from .preparation import prepare


def extract_approved(config, bundle, context, query, dataset_profile, progress=lambda *_: None):
    working = copy.deepcopy(bundle)
    working.pop('preparation', None)
    settings = working.get('research_policy')
    approved, issues = [], []
    for source in working['sources']:
        decision = source.get('dataset_review', {}).get('decision')
        if decision not in ('approve', 'independent', 'background'):
            continue
        progress('Checking source: ' + source['title'][:90])
        try:
            # Old explicit independence reviews remain usable; new simple
            # approvals use automated provenance screening when exclusions exist.
            if decision == 'approve' and screening.policy(settings)['excluded_datasets']:
                if source.get('screening_report', {}).get('key') != screening.review_key(source, settings):
                    research.screen_sources(config, [source], settings)
            screening.require_source_review(source, settings)
            approved.append(source)
        except ValueError as error:
            issues.append({'source_id': source['id'], 'reason': str(error)})
    background = [s for s in approved if s.get('dataset_review', {}).get('decision') == 'background'
                  or s.get('screening_report', {}).get('status') == 'background']
    papers = [s for s in approved if s not in background]
    drafts = []
    for paper in papers:
        progress('Extracting equations: ' + paper['title'][:90])
        try:
            result = research.extract(config, working['target'], dataset_profile,
                                      [paper, *background[:7]], settings)
            for model in result['models']:
                # Source-local identifiers prevent collisions across papers.
                original = model.get('id') or str(len(drafts))
                model['id'] = paper['id'][:10] + '-' + hashlib.sha256(original.encode()).hexdigest()[:8]
                drafts.append(model)
            issues.extend({'source_id': paper['id'], 'reason': x} for x in result.get('issues', []))
        except ValueError as error:
            issues.append({'source_id': paper['id'], 'reason': str(error)})
    working['models'] = drafts
    progress('Checking equations and available predictors')
    models, report = prepare(working, context, query)
    working['models'] = models
    report['source_issues'] = issues
    report['approved_papers'] = len(papers)
    working['preparation'] = report
    return working
