from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from .database import initialize, connect
from .schemas import PlanRequest, ExecuteRequest, SimulationRequest, ApplySimulationRequest
from .services import *
import io, json, uuid

app=FastAPI(title='DataCleanAI', version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173'],allow_methods=['*'],allow_headers=['*'])
@app.on_event('startup')
def startup(): initialize()
def fail(fn):
    try:return fn()
    except ValueError as e: raise HTTPException(400,str(e))

@app.get('/api/health')
def health(): return {'status':'ok'}
@app.post('/api/datasets/upload')
async def upload(file:UploadFile=File(...)):
    content=await file.read(); sid,df=fail(lambda:create_session(file.filename or 'upload',content))
    return {'session_id':sid,'filename':file.filename,'size':len(content),'profile':profile(df)}
@app.get('/api/datasets/{sid}/profile')
def get_profile(sid:str): return fail(lambda:profile(read_df(Path(get_session(sid)['working_path']))))
@app.get('/api/datasets/{sid}/preview')
def preview(sid:str,page:int=1,page_size:int=20):
    df=fail(lambda:read_df(Path(get_session(sid)['working_path']))); start=max(page-1,0)*min(page_size,100); part=df.iloc[start:start+min(page_size,100)].replace({np.nan:None})
    return {'columns':[str(c) for c in df.columns],'rows':part.to_dict(orient='records'),'total':len(df),'page':page}
@app.post('/api/datasets/{sid}/plan')
def plan(sid:str, request:PlanRequest):
    df=fail(lambda:read_df(Path(get_session(sid)['working_path'])))
    parsed=parse_intent(request.objective,list(df.columns), df)
    objective_ops=objective_recommendations(parsed, df)
    profile_ops=recommendations(df)
    manual_ops=[o.model_dump() for o in request.selected_operations]
    if request.all_applicable and not manual_ops:
        manual_ops=[{'name':r['name'], 'reason':r['reason']} for r in profile_ops if r['name'] in ALIASES]
    return {
        'intent': parsed,
        'objective_operations': objective_ops,
        'profile_recommendations': profile_ops,
        'manual_operations': manual_ops,
        'validation_errors': validate_operations(df, manual_ops),
    }
@app.post('/api/datasets/{sid}/clean')
def clean(sid:str, request:ExecuteRequest):
    info=fail(lambda:get_session(sid)); before=read_df(Path(info['working_path'])); ops=[o.model_dump() for o in request.operations]; errors=validate_operations(before,ops)
    destructive=any(o['name'] in ('duplicates',) or o.get('params',{}).get('strategy') in ('drop_rows','remove') for o in ops)
    if errors: raise HTTPException(400,detail={'errors':errors})
    if destructive and not request.confirm_destructive: raise HTTPException(400,'Destructive plan requires confirmation.')
    after,logs=apply_operations(before,ops); save_df(after,Path(info['working_path']))
    for log,op in zip(logs,ops): record_history(sid,log['operation'],log['parameters'],op.get('reason','User-selected operation'),before,after,log['affected_rows'],log['affected_cells'])
    warnings=[]
    if any(o['name']=='outliers' and o.get('params',{}).get('strategy','flag')=='flag' for o in ops):
        warnings.append('Outlier flagging adds _outlier_flag; statistical outliers are not necessarily errors.')
    return {'before':metrics(before),'after':metrics(after,len(before)),'log':logs,'warnings':warnings}
@app.post('/api/datasets/{sid}/simulate')
def simulate(sid:str,request:SimulationRequest):
    info=fail(lambda:get_session(sid)); original=read_df(Path(info['working_path'])); candidates=[]
    for op in [o.model_dump() for o in request.operations]:
        if op['name']=='missing': variants=['mean','median','mode','drop_rows']
        elif op['name']=='outliers': variants=['flag','cap','remove']
        elif op['name']=='categorical': variants=['trim','lower']
        else: continue
        for v in variants:
            x={**op,'params':{**op.get('params',{}),'strategy':v,'case_normalize':v=='lower'}}; result,log=apply_operations(original,[x]); m=metrics(result,len(original)); score=round(.45*m['completeness']+.25*m['record_retention']+.15*m['uniqueness']+.15*(1-min(log[0]['affected_cells']/max(original.size,1),1)),4)
            candidates.append({'candidate_id':str(uuid.uuid4()),'operation':op['name'],'strategy':v,'metrics':m,'score':score,
              'affected_cells':log[0]['affected_cells'],'affected_rows':log[0]['affected_rows'],
              'risk':'May remove records or alter values; review before applying.' if v in ('remove','drop_rows','cap') else 'Non-destructive or reversible via undo.','operation_data':x})
    if not candidates: raise HTTPException(400,'No meaningful strategy comparison for the selected operations.')
    grouped={}
    for candidate in candidates: grouped.setdefault(candidate['operation'],[]).append(candidate['score'])
    notes=[]
    for operation,scores in grouped.items():
        if max(scores)-min(scores) <= .01:
            notes.append(f'{operation}: candidate scores are tied or differ by 0.01 or less; use risk and retention rather than score alone.')
    return {'formula':'0.45 completeness + 0.25 retention + 0.15 uniqueness + 0.15 changed-cell preservation',
      'candidates':candidates, 'comparison_notes':notes}
@app.get('/api/datasets/{sid}/history')
def get_history(sid:str): return fail(lambda:history(sid))
@app.post('/api/datasets/{sid}/undo/{history_id}')
def undo(sid:str,history_id:int):
    info=fail(lambda:get_session(sid))
    with connect() as db: row=db.execute('SELECT * FROM history WHERE id=? AND session_id=?',(history_id,sid)).fetchone()
    if not row: raise HTTPException(404,'History entry not found.')
    shutil.copy2(row['snapshot_path'],info['working_path']); return {'message':'Working dataset restored to the saved pre-operation snapshot.'}
@app.get('/api/datasets/{sid}/export/{kind}')
def export(sid:str,kind:str):
    info=fail(lambda:get_session(sid)); df=read_df(Path(info['working_path']))
    if kind=='csv':
        # Prefix formula-like spreadsheet values while retaining their visible text.
        safe=df.copy()
        for col in safe.select_dtypes(include=['object', 'string']).columns:
            safe[col]=safe[col].map(lambda value: "'" + value if isinstance(value, str) and value[:1] in ('=', '+', '-', '@') else value)
        return Response(safe.to_csv(index=False),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=cleaned.csv'})
    if kind=='xlsx':
        out=io.BytesIO(); df.to_excel(out,index=False); return Response(out.getvalue(),media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':'attachment; filename=cleaned.xlsx'})
    if kind=='report': return Response(json.dumps({'session_id':sid,'metrics':metrics(df),'history':history(sid)},default=str,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=cleaning-report.json'})
    raise HTTPException(404,'Unknown export format.')
