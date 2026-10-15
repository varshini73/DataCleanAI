from io import BytesIO
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

def upload(client, name, content):
    response=client.post('/api/datasets/upload', files={'file':(name,content,'application/octet-stream')})
    assert response.status_code == 200
    return response.json()['session_id']

def test_sessions_are_isolated_and_noop_cleaning_is_safe():
    with TestClient(app) as client:
        first=upload(client,'first.csv',b'value\n1\n1\n')
        second=upload(client,'second.csv',b'value\n2\n')
        no_op=client.post(f'/api/datasets/{first}/clean',json={'operations':[{'name':'missing','params':{'strategy':'leave'}}]})
        assert no_op.status_code == 200 and no_op.json()['log'][0]['affected_cells'] == 0
        client.post(f'/api/datasets/{first}/clean',json={'operations':[{'name':'duplicates'}],'confirm_destructive':True})
        assert client.get(f'/api/datasets/{first}/preview').json()['total'] == 1
        assert client.get(f'/api/datasets/{second}/preview').json()['total'] == 1

def test_invalid_plan_fails_without_corrupting_working_dataset():
    with TestClient(app) as client:
        sid=upload(client,'values.csv',b'value\n1\n100\n')
        before=client.get(f'/api/datasets/{sid}/preview').json()
        response=client.post(f'/api/datasets/{sid}/clean',json={'operations':[{'name':'outliers','columns':['value'],'params':{'strategy':'destroy'}}]})
        assert response.status_code == 400
        assert client.get(f'/api/datasets/{sid}/preview').json() == before

def test_excel_upload_and_excel_export():
    payload=BytesIO()
    pd.DataFrame({'Marks':[1, None], 'Department':[' A ','B']}).to_excel(payload,index=False)
    with TestClient(app) as client:
        sid=upload(client,'sheet.xlsx',payload.getvalue())
        profile=client.get(f'/api/datasets/{sid}/profile').json()
        assert profile['metrics']['rows'] == 2
        exported=client.get(f'/api/datasets/{sid}/export/xlsx')
        assert exported.status_code == 200
        assert exported.headers['content-type'].startswith('application/vnd.openxmlformats')

def test_outlier_removal_requires_explicit_operation_approval():
    with TestClient(app) as client:
        sid=upload(client,'numbers.csv',b'value\n1\n2\n100\n')
        denied=client.post(f'/api/datasets/{sid}/clean',json={'operations':[{'name':'outliers','columns':['value'],'params':{'strategy':'remove'}}],'confirm_destructive':True})
        assert denied.status_code == 400
        approved=client.post(f'/api/datasets/{sid}/clean',json={'operations':[{'name':'outliers','columns':['value'],'params':{'strategy':'remove','approved':True}}],'confirm_destructive':True})
        assert approved.status_code == 200
