from fastapi.testclient import TestClient
from app.main import app

def test_health_and_invalid_upload():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==200
        r=client.post('/api/datasets/upload',files={'file':('bad.txt',b'hello','text/plain')})
        assert r.status_code==400

def test_plan_returns_objective_recommendations_without_overwriting_manual_selection():
    with TestClient(app) as client:
        uploaded=client.post(
            '/api/datasets/upload',
            files={'file': ('marks.csv', b'marks,category\n10, A \n, B\n', 'text/csv')},
        )
        assert uploaded.status_code == 200
        sid=uploaded.json()['session_id']
        response=client.post(
            f'/api/datasets/{sid}/plan',
            json={
                'objective': 'Fill missing marks and standardize categorical values.',
                'selected_operations': [{'name': 'duplicates'}],
            },
        )
        assert response.status_code == 200
        body=response.json()
        assert {item['name'] for item in body['objective_operations']} == {'missing', 'categorical'}
        assert body['manual_operations'] == [{'name': 'duplicates', 'enabled': True, 'columns': [], 'params': {}, 'reason': ''}]

def test_upload_plan_simulate_clean_export_and_undo_preserve_original():
    with TestClient(app) as client:
        payload=b'Student_ID,Marks,Department,Attendance\n1,50, CS ,90\n2,,CS,91\n2,,CS,91\n3,70,Math,10\n'
        uploaded=client.post('/api/datasets/upload',files={'file':('students.csv',payload,'text/csv')})
        assert uploaded.status_code == 200
        sid=uploaded.json()['session_id']
        original=client.get(f'/api/datasets/{sid}/preview').json()
        plan=client.post(f'/api/datasets/{sid}/plan',json={'objective':'Fill missing marks, remove duplicate rows, standardize department names, and identify attendance outliers.'})
        assert plan.status_code == 200
        by_name={item['name']:item for item in plan.json()['objective_operations']}
        assert by_name['missing']['columns']==['Marks']
        assert by_name['categorical']['columns']==['Department']
        assert by_name['outliers']['columns']==['Attendance']
        operations=[by_name[name] for name in ('missing','duplicates','categorical','outliers')]
        simulated=client.post(f'/api/datasets/{sid}/simulate',json={'operations':operations})
        assert simulated.status_code == 200
        assert client.get(f'/api/datasets/{sid}/preview').json()['rows']==original['rows']
        cleaned=client.post(f'/api/datasets/{sid}/clean',json={'operations':operations,'confirm_destructive':True})
        assert cleaned.status_code == 200
        updated=client.get(f'/api/datasets/{sid}/preview').json()
        assert '_outlier_flag' in updated['columns'] and len(updated['rows']) == 3
        exported=client.get(f'/api/datasets/{sid}/export/csv')
        assert exported.status_code == 200 and '_outlier_flag' in exported.text
        history=client.get(f'/api/datasets/{sid}/history').json()
        assert history
        assert client.post(f'/api/datasets/{sid}/undo/{history[0]["id"]}').status_code == 200
        assert client.get(f'/api/datasets/{sid}/preview').json()['rows']==original['rows']

def test_csv_export_escapes_formula_values():
    with TestClient(app) as client:
        upload=client.post('/api/datasets/upload',files={'file':('formula.csv',b'name\n"=SUM(1,1)"\n','text/csv')})
        assert upload.status_code == 200
        response=client.get(f'/api/datasets/{upload.json()["session_id"]}/export/csv')
        assert "'=SUM(1,1)" in response.text

def test_simulator_score_is_recomputed_from_each_candidate_metrics():
    with TestClient(app) as client:
        upload=client.post('/api/datasets/upload',files={'file':('values.csv',b'value\n1\n2\n100\n','text/csv')})
        sid=upload.json()['session_id']
        response=client.post(f'/api/datasets/{sid}/simulate',json={'operations':[{'name':'outliers','columns':['value'],'params':{'strategy':'flag'}}]})
        assert response.status_code == 200
        for candidate in response.json()['candidates']:
            metrics=candidate['metrics']
            expected=round(.45*metrics['completeness']+.25*metrics['record_retention']+.15*metrics['uniqueness']+.15*(1-min(candidate['affected_cells']/3,1)),4)
            assert candidate['score'] == expected

def test_profile_and_history_endpoints_reflect_current_session_state():
    with TestClient(app) as client:
        uploaded=client.post('/api/datasets/upload',files={'file':('people.csv',b'name,marks\nAda,10\nAda,10\n','text/csv')})
        sid=uploaded.json()['session_id']
        profile=client.get(f'/api/datasets/{sid}/profile')
        assert profile.status_code == 200
        assert profile.json()['metrics']['duplicate_rows'] == 1
        assert client.get(f'/api/datasets/{sid}/history').json() == []
        clean=client.post(f'/api/datasets/{sid}/clean',json={'operations':[{'name':'duplicates','reason':'Remove repeated row'}],'confirm_destructive':True})
        assert clean.status_code == 200
        history=client.get(f'/api/datasets/{sid}/history').json()
        assert len(history) == 1
        assert history[0]['operation'] == 'duplicates'
        assert history[0]['reason'] == 'Remove repeated row'
        assert history[0]['affected_rows'] == 1
