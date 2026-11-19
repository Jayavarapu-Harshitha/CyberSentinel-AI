import json,time,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .config import ROOT,config
from .models.registry import build
from .evaluation import metrics,timed_probabilities
from .fusion import blend,select_weight

def pipeline(name,seed):
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if name in ('mlp','logistic_regression'): steps.append(('scaler',StandardScaler()))
    steps.append(('model',build(name,seed)))
    return Pipeline(steps)

def train(out_dir=None,cfg=None,models=None):
    cfg=cfg or config(); out=Path(out_dir or ROOT/'artifacts');out.mkdir(exist_ok=True,parents=True)
    data={}
    for split in ('train','val','test'):
        df=pd.read_pickle(out/f'{split}.pkl');data[split]=(df.drop(columns='_target'),df['_target'])
    trained={}; validation={}; errors={}; results=[]
    for name in (models or cfg['models']):
        try:
            model=pipeline(name,cfg['seed']);start=time.perf_counter()
            model.fit(*data['train']);train_seconds=time.perf_counter()-start
            val_p=model.predict_proba(data['val'][0])[:,1]
            validation[name]=metrics(data['val'][1],val_p,cfg['threshold'])
            trained[name]=(model,train_seconds,val_p)
            joblib.dump(model,out/f'{name}.joblib')
        except (ImportError,ModuleNotFoundError,OSError) as e:
            errors[name]=f'{type(e).__name__}: {e}'
    if not trained: raise RuntimeError('No models trained: '+str(errors))
    for name,(model,train_seconds,_) in trained.items():
        p,latency=timed_probabilities(model,data['test'][0]);r=metrics(data['test'][1],p,cfg['threshold'])
        results.append({'model':name,'train_seconds':train_seconds,**latency,**r})
    fusion_info=None
    if 'mlp' in trained:
        trees=[n for n in ('logistic_regression','random_forest','hist_gradient_boosting','xgboost','lightgbm') if n in trained]
        if trees:
            # Pick tree by validation PR-AUC, not test performance.
            tree=max(trees,key=lambda n:validation[n]['pr_auc'])
            p1=trained['mlp'][2];p2=trained[tree][2]
            weight=select_weight(data['val'][1],p1,p2)
            fusion_info={'tree':tree,'mlp_weight':weight,'selection':'tree by validation PR-AUC; fusion weight by validation F1'}
            for label,w in [('equal_fusion',.5),('tuned_fusion',weight)]:
                t=time.perf_counter()
                pt=blend(trained['mlp'][0].predict_proba(data['test'][0])[:,1],trained[tree][0].predict_proba(data['test'][0])[:,1],w)
                seconds=time.perf_counter()-t
                results.append({'model':label,'train_seconds':trained['mlp'][1]+trained[tree][1], 'inference_seconds_median':None,'inference_ms_per_row_median':None,'fusion_single_run_seconds':seconds,**metrics(data['test'][1],pt,cfg['threshold'])})
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    (out/'training_summary.json').write_text(json.dumps({'skipped_models':errors,'fusion':fusion_info,'warning':'Random stratified sampled-data evaluation; not full dataset or unseen-day evaluation'},indent=2))
    return results
