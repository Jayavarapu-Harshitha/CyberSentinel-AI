"""Optional anomaly detection, feature explanations, ablations and benchmarking.
All fitting/selection uses training/validation data only.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
from .evaluation import metrics, timed_probabilities
from .fusion import blend
from .training import pipeline


def read_partitions(out):
    result = {}
    for name in ('train', 'val', 'test'):
        frame = pd.read_pickle(Path(out) / f'{name}.pkl')
        result[name] = (frame.drop(columns='_target'), frame['_target'].astype(int))
    return result


def anomaly_experiment(out, seed=42, max_normal=15000):
    """Fit anomaly model ONLY on benign training examples; no supervised attack labels."""
    out = Path(out); data = read_partitions(out)
    xtrain, ytrain = data['train']; xval, yval = data['val']; xtest, ytest = data['test']
    normal = xtrain.loc[ytrain.eq(0)]
    if len(normal) > max_normal: normal = normal.sample(max_normal, random_state=seed)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    a = imp.fit_transform(normal)
    model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=2, random_state=seed)
    start = time.perf_counter(); model.fit(a); train_seconds = time.perf_counter()-start
    # Larger scores mean more anomalous; percentile threshold from NORMAL training only.
    normal_scores = -model.score_samples(a)
    threshold = float(np.quantile(normal_scores, .95))
    scores = -model.score_samples(imp.transform(xtest))
    predictions = (scores >= threshold).astype(int)
    summary = {'method':'unsupervised IsolationForest trained on benign train only', 'train_seconds':train_seconds,
               'threshold':'95th percentile of benign TRAIN anomaly scores', 'threshold_value':threshold,
               'test_metrics':metrics(ytest, predictions.astype(float), threshold=.5),
               'test_average_precision_anomaly_scores':float(average_precision_score(ytest, scores)) if ytest.nunique()==2 else None,
               'test_roc_auc_anomaly_scores':float(roc_auc_score(ytest, scores)) if ytest.nunique()==2 else None}
    joblib.dump({'imputer':imp,'model':model,'threshold':threshold},out/'isolation_forest.joblib')
    (out/'anomaly.json').write_text(json.dumps(summary,indent=2))
    return summary


def feature_explanation(out, max_rows=150, seed=42):
    """Model-agnostic permutation importance; optional SHAP for tree model.
    Importance uses held-out validation subset, not final test, to avoid test-set exploration.
    """
    out=Path(out); data=read_partitions(out)
    info=json.loads((out/'training_summary.json').read_text())
    tree=(info.get('fusion') or {}).get('tree')
    if tree is None: raise ValueError('Train MLP and a tree model first')
    model=joblib.load(out/f'{tree}.joblib')
    x,y=data['val']; n=min(len(x),max_rows)
    if n<10: raise ValueError('Need at least 10 validation records')
    ix=np.random.default_rng(seed).choice(len(x),size=n,replace=False)
    xv=x.iloc[ix];yv=y.iloc[ix]
    imp=permutation_importance(model,xv,yv,n_repeats=3,random_state=seed,scoring='average_precision',n_jobs=1)
    features=sorted([{'feature':str(col),'importance_mean':float(v),'importance_std':float(s)} for col,v,s in zip(x.columns,imp.importances_mean,imp.importances_std)],key=lambda a:a['importance_mean'],reverse=True)
    result={'tree':tree,'method':'permutation importance on sampled validation rows','validation_rows':n,'top_features':features[:20]}
    (out/'feature_importance.json').write_text(json.dumps(result,indent=2))
    return result


def ablation_experiment(out, seed=42):
    """Compare full-feature MLP, tree, equal and tuned fusion, plus reduced-feature MLP.
    Feature selection is derived only from training statistics; test evaluated once.
    """
    out=Path(out); data=read_partitions(out)
    info=json.loads((out/'training_summary.json').read_text())
    fusion=info.get('fusion')
    if not fusion: raise ValueError('Run full training with MLP and tree first')
    xt,yt=data['test']; xv,yv=data['val']; xtr,ytr=data['train']
    mlp=joblib.load(out/'mlp.joblib');tree=joblib.load(out/f"{fusion['tree']}.joblib")
    p1=mlp.predict_proba(xt)[:,1];p2=tree.predict_proba(xt)[:,1]
    cases={'mlp_only':p1,'tree_only':p2,'equal_fusion':blend(p1,p2,.5),'tuned_fusion':blend(p1,p2,fusion['mlp_weight'])}
    # Feature ablation: choose half the columns by training-only variance after numeric coercion.
    numeric=xtr.apply(pd.to_numeric,errors='coerce')
    variance=numeric.var(skipna=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    kept=variance.sort_values(ascending=False).head(max(1,len(variance)//2)).index.tolist()
    reduced=pipeline('mlp',seed)
    start=time.perf_counter();reduced.fit(xtr[kept],ytr);seconds=time.perf_counter()-start
    cases['mlp_half_features']=reduced.predict_proba(xt[kept])[:,1]
    result={'feature_ablation':'half features by training-set variance; exploratory, not optimal feature selection',
            'reduced_feature_count':len(kept),'reduced_training_seconds':seconds,
            'test_metrics':{name:metrics(yt,p) for name,p in cases.items()}}
    (out/'ablation.json').write_text(json.dumps(result,indent=2))
    return result


def latency_experiment(out,repeats=7):
    out=Path(out);data=read_partitions(out);x,_=data['test']
    info=json.loads((out/'training_summary.json').read_text()); f=info.get('fusion')
    names=['logistic_regression','random_forest','xgboost','lightgbm','mlp','hist_gradient_boosting']
    result={}
    for name in names:
        path=out/f'{name}.joblib'
        if path.exists():
            _,t=timed_probabilities(joblib.load(path),x,repeats=repeats)
            result[name]=t
    if f:
        a=joblib.load(out/'mlp.joblib');b=joblib.load(out/f"{f['tree']}.joblib")
        class Fused:
            def predict_proba(self,x):
                p=blend(a.predict_proba(x)[:,1],b.predict_proba(x)[:,1],f['mlp_weight'])
                return np.column_stack((1-p,p))
        _,result['tuned_fusion']=timed_probabilities(Fused(),x,repeats=repeats)
    (out/'latency.json').write_text(json.dumps({'repeats':repeats,'measurements':result},indent=2))
    return result
