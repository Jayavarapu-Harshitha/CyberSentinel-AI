"""Optional local FastAPI inference endpoint; do not expose publicly without auth/rate limits."""
from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from cybersentinel.fusion import blend
app=FastAPI(title='CyberSentinel AI',version='0.2.0')
class Batch(BaseModel):
    records:list[dict[str,float|int|None]]=Field(min_length=1,max_length=100)
@app.get('/health')
def health(): return {'status':'ok','models_available':Path('artifacts/training_summary.json').exists()}
@app.post('/predict')
def predict(batch:Batch):
    out=Path('artifacts')
    if not (out/'training_summary.json').exists(): raise HTTPException(503,'Models not trained')
    f=json.loads((out/'training_summary.json').read_text()).get('fusion')
    if not f: raise HTTPException(503,'Fusion model not trained')
    mlp=joblib.load(out/'mlp.joblib');tree=joblib.load(out/f"{f['tree']}.joblib")
    x=pd.DataFrame(batch.records);required=list(mlp.feature_names_in_)
    missing=[c for c in required if c not in x]
    if missing: raise HTTPException(422,{'missing_features':missing})
    x=x[required].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    p=blend(mlp.predict_proba(x)[:,1],tree.predict_proba(x)[:,1],f['mlp_weight'])
    return {'results':[{'malicious_score':float(v),'flagged_for_review':bool(v>=.5)} for v in p],'disclaimer':'Research model; prediction is not proof of an attack'}
