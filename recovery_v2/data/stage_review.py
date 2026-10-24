"""Stage review candidates; uncertain overlap is excluded, never auto-approved."""
import json
import re
from collections import Counter
from asi.data.recovery_corpus import BIOLOGY
from recovery_v2.data.prepare_candidates import ROOT, sha, normalize
from recovery_v2.preflight import save


def signature(value):
    return re.sub(r'\d+(?:\s*\.\s*\d+)?', '<n>', normalize(value.split('\n')[0]))


def main():
    base = ROOT/'results/v2/F0'
    paths = [base/'data_audit/candidates.json', base/'inventory_consolidated/candidate_review.json',
             base/'self_cache_resolution/report.json', base/'exposure_inventory/identified_exposure.json']
    candidates, review, resolved, historical = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    states = {r['id']:r['status'] for r in review['records']}
    self_cache = set(resolved['resolved_ids'])
    old_templates = {signature(r['record']['prompt']) for r in historical['records']}
    kept, exclusions, reasons = [], [], Counter()
    for item in candidates['examples']:
        row = item['row']; question = row.get('question',row.get('text',''))
        reason = None
        if states[item['id']] == 'recorded_evaluation_overlap': reason = 'recorded_evaluation_overlap'
        elif states[item['id']] == 'cached_overlap_unresolved' and item['id'] not in self_cache:
            reason = 'uncertain_provenance_excluded'
        elif signature(question) in old_templates: reason = 'historical_numeric_template'
        elif item['source'] == 'sciq' and not re.search(BIOLOGY, question+' '+row.get('support',''), re.I):
            reason = 'outside_lexical_biology_filter'
        elif item['source'] == 'gsm8k' and not re.search(r'####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$',row['answer']):
            reason = 'missing_numeric_answer'
        if reason:
            reasons[reason] += 1
            exclusions.append({'id':item['id'],'reason':reason})
        else:
            kept.append({**item,'review_status':'pending','category_provisional':
                         {'gsm8k':'math','mbpp':'code','sciq':'bio'}[item['source']]})
    out = base/'review_staging'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists(): raise SystemExit('Existing review preserved.')
    save(out/'candidates.json',{'examples':kept,'split_assignment':None})
    save(out/'exclusions.json',{'records':exclusions})
    save(out/'report.json',{'status':'partial','counts':dict(Counter(r['source'] for r in kept)),
                            'exclusions':dict(reasons),'training_authorized':False,'evaluation_authorized':False,
                            'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
                            'outputs_sha256':{name:sha(out/name) for name in ('candidates.json','exclusions.json')},
                            'script_sha256':sha(__import__('pathlib').Path(__file__)),
                            'limitations':['Lexical domain/template checks do not constitute semantic review.',
                                           'References have not been verified; no splits frozen.',
                                           'All uncertain cache provenance excluded conservatively.']})


if __name__ == '__main__': main()
