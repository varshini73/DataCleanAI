"""Local TF-IDF candidate matching with deterministic rule validation."""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .intent_catalog import INTENT_EXAMPLES

MIN_SIMILARITY = 0.20
NEGATION_RE = re.compile(r"\b(?:do not|don't|dont|only show|only detect|without removing|keep)\b", re.I)

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s-]', ' ', text.lower())).strip()

@lru_cache(maxsize=1)
def model() -> tuple[TfidfVectorizer, list[str], list[str]]:
    labels, examples = [], []
    for label, phrases in INTENT_EXAMPLES.items():
        labels.extend([label] * len(phrases))
        examples.extend(phrases)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', sublinear_tf=True)
    vectorizer.fit(examples)
    return vectorizer, labels, examples

def clauses(text: str) -> list[str]:
    cleaned = re.sub(r'\b(?:and|then|also)\b', '|', text, flags=re.I)
    return [part.strip() for part in re.split(r'[,;|.]', cleaned) if part.strip()]

def _rule_candidates(clause: str) -> set[str]:
    low = clause.lower()
    found: set[str] = set()
    if re.search(r'\b(?:missing|null|nan|blank|imput\w*|fill|absent|empty)\b', low): found.add('missing')
    if re.search(r'\b(?:duplicate\w*|dedup\w*|repeated|repeat\w*)\b', low): found.add('duplicates')
    if re.search(r'\b(?:outlier\w*|anomal\w*|unusual|extreme|strange)\b', low): found.add('outliers')
    if re.search(r'\b(?:standardi[sz]\w*|categor\w*|trim|whitespace|capitali[sz]\w*)\b', low): found.add('categorical')
    if re.search(r'\b(?:convert|numeric|datetime|date|data type)\b', low): found.add('convert')
    if re.search(r'\b(?:scale|normaliz|machine learning|\bml\b)\b', low): found.add('scale')
    if re.search(r'\b(?:profile|inspect|quality summary|quality problems|quality overview)\b', low): found.add('profile')
    if re.search(r'\b(?:export|download)\b', low): found.add('export')
    if re.search(r'\b(?:undo|restore)\b', low): found.add('undo')
    return found

def rule_baseline(text: str) -> set[str]:
    """Keyword-only baseline used only for reproducible evaluation."""
    return {label for clause in clauses(text) for label in _rule_candidates(normalize_text(clause))}

def tfidf_baseline(text: str) -> set[str]:
    """Nearest TF-IDF example per clause, rejecting weak lexical matches."""
    if not normalize_text(text):
        return set()
    vectorizer, labels, examples = model()
    predicted: set[str] = set()
    matrix = vectorizer.transform(examples)
    for raw_clause in clauses(text):
        scores = cosine_similarity(vectorizer.transform([normalize_text(raw_clause)]), matrix)[0]
        index = int(scores.argmax())
        if scores[index] >= MIN_SIMILARITY:
            predicted.add(labels[index])
    return predicted

def recognize(text: str) -> list[dict[str, Any]]:
    if not normalize_text(text):
        return []
    vectorizer, labels, examples = model()
    results: list[dict[str, Any]] = []
    for raw_clause in clauses(text):
        clause = normalize_text(raw_clause)
        vector = vectorizer.transform([clause])
        scores = cosine_similarity(vector, vectorizer.transform(examples))[0]
        best_by_label: dict[str, tuple[float, str]] = {}
        for label, score, example in zip(labels, scores, examples):
            if score > best_by_label.get(label, (-1.0, ''))[0]:
                best_by_label[label] = (float(score), example)
        for label in _rule_candidates(clause):
            score, example = best_by_label.get(label, (0.0, 'rule vocabulary'))
            results.append({'name': label, 'similarity': round(score, 3), 'matched_example': example,
                'clause': clause, 'negated': bool(NEGATION_RE.search(clause)),
                'recognition': 'tfidf+rule' if score >= MIN_SIMILARITY else 'rule fallback'})
    unique: dict[str, dict[str, Any]] = {}
    for item in results:
        previous = unique.get(item['name'])
        if previous is None or item['similarity'] > previous['similarity']:
            unique[item['name']] = item
    return list(unique.values())
