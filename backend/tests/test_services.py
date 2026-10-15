import pandas as pd
from app.services import metrics, parse_intent, apply_operations, objective_recommendations, validate_operations

def test_metrics_and_missing_imputation():
    df=pd.DataFrame({'marks':[10,None,30],'kind':[' A ','B','B']})
    cleaned, log=apply_operations(df,[{'name':'missing','params':{'strategy':'mean'},'columns':['marks']},{'name':'categorical','params':{},'columns':['kind']}])
    assert cleaned.marks.isna().sum()==0 and cleaned.kind.iloc[0]=='A' and log[0]['affected_cells']>0

def test_duplicates_and_outliers_do_not_mutate_source():
    df=pd.DataFrame({'a':[1,1,2,100]})
    result,_=apply_operations(df,[{'name':'duplicates','params':{}},{'name':'outliers','params':{'strategy':'flag'},'columns':['a']}])
    assert len(df)==4 and '_outlier_flag' in result

def test_intent_and_validation():
    parsed=parse_intent('Remove duplicate rows and fill missing marks',['marks'])
    assert {x['name'] for x in parsed['operations']} == {'duplicates','missing'}
    assert validate_operations(pd.DataFrame({'a':[1]}),[{'name':'scale','columns':['bad'],'params':{}}])

def test_intent_parser_recognizes_supported_objectives_and_columns():
    parsed=parse_intent(
        'Remove duplicate records, fill missing marks, standardize categorical values, and identify statistical outliers.',
        ['marks', 'category'],
    )
    assert {item['name'] for item in parsed['operations']} == {'duplicates', 'missing', 'categorical', 'outliers'}
    missing=next(item for item in parsed['operations'] if item['name']=='missing')
    assert missing['columns'] == ['marks']
    proposals=objective_recommendations(parsed, pd.DataFrame({'marks':[1, None], 'category':[' A ', 'B']}))
    outliers=next(item for item in proposals if item['name']=='outliers')
    assert outliers['params']['strategy'] == 'flag'
    assert '_outlier_flag' in outliers['note']

def test_intent_parser_explains_empty_and_unsupported_objectives():
    empty=parse_intent('', ['marks'])
    unsupported=parse_intent('Predict student grades perfectly', ['marks'])
    assert empty['operations'] == [] and empty['assumptions']
    assert unsupported['operations'] == [] and unsupported['unsupported']

def test_dataset_aware_multi_intent_does_not_assign_marks_to_every_operation():
    df=pd.DataFrame({
        'Student_ID':[101, 102, 103, 104],
        'Marks':[50, None, 70, 80],
        'Department':[' CS ', 'CS', 'Math', 'Math'],
        'Attendance_Percent':[90, 91, 89, 10],
    })
    parsed=parse_intent(
        'Fill missing marks, standardize department names, and identify attendance outliers.',
        list(df.columns), df,
    )
    operations={item['name']: item for item in parsed['operations']}
    assert operations['missing']['columns'] == ['Marks']
    assert operations['categorical']['columns'] == ['Department']
    assert operations['outliers']['columns'] == ['Attendance_Percent']
    assert 'Student_ID' not in operations['outliers']['columns']

def test_negated_duplicate_request_is_not_an_executable_removal():
    df=pd.DataFrame({'Marks':[1, 1]})
    parsed=parse_intent('Do not remove duplicates; only show them.', list(df.columns), df)
    assert parsed['operations'][0]['name'] == 'duplicates'
    assert parsed['operations'][0]['negated'] is True
    assert objective_recommendations(parsed, df) == []
