from pathlib import Path
import pandas as pd

out=Path(__file__).resolve().parents[1]/'example_students.csv'
pd.DataFrame({'student':[' Ada ','Ben','Ben','Cara'], 'marks':[84,None,None,999], 'date':['2025-01-01','bad','bad','2025-03-01']}).to_csv(out,index=False)
print(out)
