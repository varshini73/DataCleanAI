"""Small, local training catalog for DataCleanAI's lexical intent matcher."""

# Examples are distinct from held-out examples in scripts/evaluate_nlp.py.
INTENT_EXAMPLES: dict[str, list[str]] = {
    'missing': ['fill missing values', 'impute null values', 'replace blank entries', 'complete absent marks', 'use median for missing numbers'],
    'duplicates': ['remove duplicate rows', 'delete repeated records', 'deduplicate the dataset', 'find duplicate records only', 'show repeated rows do not delete them'],
    'categorical': ['standardize category values', 'trim extra spaces in departments', 'normalize category capitalization', 'clean inconsistent text categories'],
    'outliers': ['detect numerical outliers', 'flag unusual values', 'identify anomalies', 'remove outlier records', 'find statistical extremes'],
    'convert': ['convert values to numeric', 'parse date column', 'convert a field to datetime', 'change data type'],
    'scale': ['scale numerical features', 'normalize numeric columns', 'standardize features for machine learning'],
    'profile': ['inspect the dataset', 'show data quality problems', 'generate quality summary', 'profile this data only'],
    'export': ['export cleaned data', 'download cleaned dataset'],
    'undo': ['undo the last cleaning operation', 'restore previous dataset version'],
}

EXECUTABLE_INTENTS = {'missing', 'duplicates', 'categorical', 'outliers', 'convert', 'scale'}
