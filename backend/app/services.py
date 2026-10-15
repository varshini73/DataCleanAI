"""Pure, deterministic data handling services used by API routes and tests."""
from __future__ import annotations
import json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .config import DATA_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from .database import connect

def now(): return datetime.now(timezone.utc).isoformat()
def session_dir(sid: str) -> Path:
    path = DATA_DIR / sid; path.mkdir(parents=True, exist_ok=True); return path
def path_for(sid: str, name: str) -> Path: return session_dir(sid) / name

def load_dataset(path: Path, original_name: str | None = None) -> pd.DataFrame:
    ext = (original_name or path.name).lower()
    try:
        if ext.endswith('.csv'): df = pd.read_csv(path)
        elif ext.endswith(('.xlsx', '.xls')): df = pd.read_excel(path)
        elif ext.endswith('.json'):
            raw = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(raw, dict): raw = raw.get('records', raw.get('data', raw))
            if not isinstance(raw, list): raise ValueError('JSON must be an array of records or contain records/data.')
            df = pd.json_normalize(raw)
        else: raise ValueError('Unsupported format.')
    except Exception as exc: raise ValueError(f'Could not read dataset: {exc}') from None
    if df.empty and len(df.columns) == 0: raise ValueError('Dataset has no rows or columns.')
    return df

def save_df(df: pd.DataFrame, path: Path): df.to_pickle(path)
def read_df(path: Path) -> pd.DataFrame: return pd.read_pickle(path)
def safe_session(sid: str):
    if not re.fullmatch(r'[a-f0-9-]{36}', sid): raise ValueError('Invalid session identifier.')

def metrics(df: pd.DataFrame, base_rows: int | None = None) -> dict[str, Any]:
    cells = max(df.shape[0] * df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    rows = len(df)
    return {'rows': rows, 'columns': len(df.columns), 'missing_cells': missing,
            'duplicate_rows': duplicates, 'completeness': round((cells-missing)/cells, 4),
            'uniqueness': round((rows-duplicates)/max(rows, 1), 4),
            'record_retention': round(rows/max(base_rows or rows, 1), 4)}

def profile(df: pd.DataFrame) -> dict[str, Any]:
    cols=[]
    for c in df.columns:
        s=df[c]; examples=[]
        if s.dtype == object:
            examples=[str(x) for x in s.dropna().head(3) if str(x).strip()!=str(x)]
        cols.append({'name': str(c), 'dtype': str(s.dtype), 'unique': int(s.nunique(dropna=True)),
          'missing': int(s.isna().sum()), 'sample_values': [str(v) for v in s.dropna().head(4).tolist()],
          'suspicious': 'leading/trailing whitespace detected' if examples else None,
          'statistics': ({k: (None if pd.isna(v) else float(v)) for k,v in s.describe().to_dict().items()} if pd.api.types.is_numeric_dtype(s) else {})})
    return {'metrics': metrics(df), 'columns': cols}

ALIASES={'missing':['missing','null','nan','fill','impute'], 'duplicates':['duplicate','dedupe','repeated'],
 'outliers':['outlier','anomaly','anomalies'], 'convert':['convert','numeric','datetime','date'],
 'categorical':['standardize','standardise','trim','categorical','case'], 'scale':['scale','normaliz','machine learning','ml']}
def parse_intent(text: str, columns: list[str]) -> dict[str, Any]:
    low=text.lower(); found=[]
    for op, words in ALIASES.items():
        if any(w in low for w in words):
            matched=[c for c in columns if c.lower() in low]
            found.append({'name':op,'columns':matched,'confidence':'high' if matched else 'medium',
              'reason':f'Matched supported {op} vocabulary.'})
    unsupported=[]
    assumptions=[]
    if not text.strip():
        assumptions.append('No objective was supplied, so no objective-based operations were recommended.')
    elif not found:
        unsupported.append('No supported preprocessing request was recognized. Use terms such as missing, duplicate, outlier, convert, standardize, or scale.')
    if any(not x['columns'] for x in found):
        assumptions.append('No column was identified for one or more operations; the operation will use its safe applicable defaults unless you choose columns.')
    return {'operations':found,'unsupported':unsupported,'assumptions':assumptions}

DEFAULT_OPERATION_PARAMS = {
    'missing': {'strategy': 'median'},
    'duplicates': {},
    'outliers': {'strategy': 'flag', 'method': 'iqr'},
    'convert': {'kind': 'numeric'},
    'categorical': {'case_normalize': False},
    'scale': {'strategy': 'standard'},
}

def objective_recommendations(intent: dict[str, Any], df: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn recognized intent into reviewable, non-executing operation proposals."""
    proposals = []
    for requested in intent['operations']:
        name = requested['name']
        columns = requested['columns']
        applicable_columns = columns or list(df.columns)
        proposals.append({
            'name': name,
            'columns': columns,
            'params': DEFAULT_OPERATION_PARAMS[name].copy(),
            'reason': requested['reason'],
            'confidence': requested['confidence'],
            'source': 'objective',
            'note': (
                'Flagging adds a boolean _outlier_flag column to the cleaned dataset, export, and preview; it does not remove records.'
                if name == 'outliers' else
                'No column was named; review the operation before execution.' if not columns else
                'Columns matched from the objective: ' + ', '.join(applicable_columns)
            ),
        })
    return proposals

def recommendations(df: pd.DataFrame) -> list[dict]:
    out=[]
    if df.isna().any().any(): out.append({'name':'missing','reason':'Dataset contains missing cells.'})
    if df.duplicated().any(): out.append({'name':'duplicates','reason':'Dataset contains duplicate records.'})
    if any(df[c].dtype == object and df[c].astype(str).str.strip().ne(df[c].astype(str)).any() for c in df.columns): out.append({'name':'categorical','reason':'Text values contain surrounding whitespace.'})
    if df.select_dtypes(include=np.number).shape[1]: out.append({'name':'outliers','reason':'Numerical columns can be inspected for statistical outliers.'})
    return out

def validate_operations(df: pd.DataFrame, operations: list[dict]) -> list[str]:
    errors=[]; allowed=set(ALIASES)
    for op in operations:
        if op['name'] not in allowed: errors.append('Operation is not allowed.')
        if any(c not in df.columns for c in op.get('columns', [])): errors.append(f"Unknown column in {op['name']}.")
        if op['name']=='outliers' and op.get('params',{}).get('strategy','flag')=='remove' and not op.get('params',{}).get('approved',False): errors.append('Outlier removal requires approved=true.')
    return errors

def changed_cell_count(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Count cells that changed across an operation, including schema/row changes.

    Both axes are aligned to their union. A dedicated sentinel makes two missing
    values compare equal while treating a missing value versus an added value as
    a change. This avoids DataFrame's same-label comparison requirement.
    """
    index = before.index.union(after.index, sort=False)
    columns = before.columns.union(after.columns, sort=False)
    sentinel = object()
    left = before.reindex(index=index, columns=columns).astype(object)
    right = after.reindex(index=index, columns=columns).astype(object)
    left = left.where(left.notna(), sentinel)
    right = right.where(right.notna(), sentinel)
    return int(left.ne(right).to_numpy().sum())

def apply_operations(source: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame,list[dict]]:
    df=source.copy(deep=True); log=[]
    for op in operations:
        if not op.get('enabled',True): continue
        name=op['name']; p=op.get('params',{}); cols=op.get('columns') or list(df.columns); before=df.copy(deep=True); affected=0
        if name=='missing':
            strategy=p.get('strategy','leave'); target=[c for c in cols if c in df]
            if strategy=='drop_rows': df=df.dropna(subset=target)
            elif strategy in ('mean','median'):
                for c in target:
                    if pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(getattr(df[c],strategy)())
            elif strategy=='mode':
                for c in target:
                    mode=df[c].mode(dropna=True)
                    if not mode.empty: df[c]=df[c].fillna(mode.iloc[0])
        elif name=='duplicates': df=df.drop_duplicates()
        elif name=='categorical':
            for c in cols:
                if c in df and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
                    df[c]=df[c].map(lambda v: v.strip() if isinstance(v,str) else v)
                    if p.get('case_normalize'): df[c]=df[c].map(lambda v: v.lower() if isinstance(v,str) else v)
        elif name=='convert':
            kind=p.get('kind','numeric')
            for c in cols:
                if c in df: df[c]=pd.to_numeric(df[c],errors='coerce') if kind=='numeric' else pd.to_datetime(df[c],errors='coerce') if kind=='datetime' else df[c].astype('category')
        elif name=='scale':
            target=[c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c])]
            if target:
                scaler=StandardScaler() if p.get('strategy','standard')=='standard' else MinMaxScaler(); df[target]=scaler.fit_transform(df[target])
        elif name=='outliers':
            target=[c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c])]; mask=pd.Series(False,index=df.index)
            for c in target:
                if p.get('method','iqr')=='zscore':
                    sd=df[c].std(); mask |= ((df[c]-df[c].mean()).abs()>float(p.get('threshold',3))*sd) if sd else False
                else:
                    q1,q3=df[c].quantile(.25),df[c].quantile(.75); iqr=q3-q1; mask |= (df[c]<q1-1.5*iqr)|(df[c]>q3+1.5*iqr)
            strategy=p.get('strategy','flag')
            if strategy=='flag': df['_outlier_flag']=mask
            elif strategy=='remove': df=df.loc[~mask]
            elif strategy=='cap':
                for c in target: df[c]=df[c].clip(df[c].quantile(.01),df[c].quantile(.99))
        affected = changed_cell_count(before, df)
        log.append({'operation':name,'affected_cells':affected,'affected_rows':abs(len(before)-len(df)),'parameters':p})
    return df,log

def create_session(filename: str, content: bytes) -> tuple[str, pd.DataFrame]:
    if len(content)>MAX_UPLOAD_BYTES: raise ValueError('File exceeds configured upload limit.')
    ext=Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS: raise ValueError('Unsupported file extension.')
    sid=str(uuid.uuid4()); raw=path_for(sid,'upload'+ext); raw.write_bytes(content)
    df=load_dataset(raw, filename); orig=path_for(sid,'original.pkl'); work=path_for(sid,'working.pkl'); save_df(df,orig); save_df(df,work)
    with connect() as db: db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?)',(sid,Path(filename).name,str(orig),str(work),now(),len(df),len(df.columns)))
    return sid,df

def get_session(sid: str):
    safe_session(sid)
    with connect() as db: row=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
    if not row: raise ValueError('Session not found.')
    return dict(row)

def record_history(sid, op, params, reason, before, after, affected_rows=0, affected_cells=0):
    snap=path_for(sid,f'snapshot-{uuid.uuid4()}.pkl'); save_df(before,snap)
    with connect() as db: db.execute('INSERT INTO history(session_id,created_at,operation,parameters,status,reason,affected_rows,affected_cells,before_metrics,after_metrics,snapshot_path) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(sid,now(),op,json.dumps(params), 'completed',reason,affected_rows,affected_cells,json.dumps(metrics(before)),json.dumps(metrics(after,len(before))),str(snap)))

def history(sid):
    with connect() as db: return [dict(r) for r in db.execute('SELECT * FROM history WHERE session_id=? ORDER BY id DESC',(sid,))]
