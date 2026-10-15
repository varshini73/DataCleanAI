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
