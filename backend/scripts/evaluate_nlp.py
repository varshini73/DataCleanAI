"""Held-out comparison of keyword, TF-IDF, and combined local matching.

Run from backend: .\.venv\Scripts\python.exe -m scripts.evaluate_nlp
This is a small regression fixture, not a generalization benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
from app.nlp import recognize, rule_baseline, tfidf_baseline

CASES = [
    ('replace absent grades using the middle value', {'missing'}), ('complete empty attendance cells', {'missing'}),
    ('purge repeated enrolments', {'duplicates'}), ('show repeated entries but keep all rows', {'duplicates'}),
    ('strip blanks around faculty labels', {'categorical'}), ('make course names use consistent letter case', {'categorical'}),
    ('find strange attendance measurements', {'outliers'}), ('remove extreme scores after review', {'outliers'}),
    ('turn admission_day into a date', {'convert'}), ('make fee values numeric', {'convert'}),
    ('rescale continuous predictors for an ML experiment', {'scale'}), ('inspect this table for quality issues', {'profile'}),
    ('save the cleaned table for download', {'export'}), ('revert the latest transformation', {'undo'}),
    ('fill blank Marks and remove repeated student records', {'missing', 'duplicates'}),
    ('clean Department labels and flag unusual Attendance_Percent values', {'categorical', 'outliers'}),
    ('parse Join_Date then normalize exam scores', {'convert', 'scale'}),
    ('do not delete duplicates; only detect them', {'duplicates'}),
    ('do not remove anomalous readings, flag them instead', {'outliers'}),
    ('compose a haiku about spreadsheets', set()), ('forecast next semester grades', set()),
    ('colour the dashboard purple', set()), ('handle values in the nonexistent column', set()),
    ('median impute marks and tidy department', {'missing', 'categorical'}),
    ('dedupe, inspect quality, and export the result', {'duplicates', 'profile', 'export'}),
]
INTENTS = sorted({label for _, labels in CASES for label in labels})

def _prf(values: dict[str, int]) -> dict[str, float]:
    p=values['tp']/(values['tp']+values['fp']) if values['tp']+values['fp'] else 0
    r=values['tp']/(values['tp']+values['fn']) if values['tp']+values['fn'] else 0
    return {'precision':round(p,3),'recall':round(r,3),'f1':round(2*p*r/(p+r),3) if p+r else 0}

def scores(rows: list[tuple[set[str], set[str]]]) -> dict:
    tp=fp=fn=exact=unknown_total=unknown_correct=0
    per={label:{'tp':0,'fp':0,'fn':0} for label in INTENTS}
    for expected, predicted in rows:
        tp += len(expected & predicted); fp += len(predicted - expected); fn += len(expected - predicted); exact += expected == predicted
        if not expected: unknown_total += 1; unknown_correct += not predicted
        for label in INTENTS:
            per[label]['tp'] += label in expected and label in predicted
            per[label]['fp'] += label not in expected and label in predicted
            per[label]['fn'] += label in expected and label not in predicted
    detailed={label:_prf(values) for label,values in per.items()}
    return {'case_count':len(rows),'micro':_prf({'tp':tp,'fp':fp,'fn':fn}),
      'macro_f1':round(sum(item['f1'] for item in detailed.values())/len(detailed),3),
      'per_intent':detailed,'exact_match_accuracy':round(exact/len(rows),3),
      'unknown_rejection_accuracy':round(unknown_correct/unknown_total,3)}

def main() -> None:
    methods={'rule_baseline':rule_baseline,'tfidf_baseline':tfidf_baseline,
      'combined_tfidf_rule':lambda text:{item['name'] for item in recognize(text)}}
    result={'methodology':'Same 25 independently phrased held-out cases for every method; cosine similarity is not a probability.',
      'limitations':'Small manually authored fixture; no claim of broad language understanding.', 'results':{}, 'cases':[]}
    for name,predict in methods.items():
        result['results'][name]=scores([(expected,predict(text)) for text,expected in CASES])
    for text,expected in CASES:
        result['cases'].append({'text':text,'expected':sorted(expected),**{name:sorted(predict(text)) for name,predict in methods.items()}})
    Path(__file__).with_name('nlp_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
