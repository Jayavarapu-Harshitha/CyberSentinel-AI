"""Unseen-capture-file evaluation (not unseen attack family generalization)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import load_samples,normalize_columns,DROP
from .config import ROOT,config
from .training import pipeline
from .evaluation import metrics

def run_day_holdout(zip_path, out_dir=None, holdout_contains='Friday-WorkingHours-Afternoon-DDos', models=('mlp','random_forest'),seed=42):
    cfg=config();out=Path(out_dir or ROOT/'artifacts');out.mkdir(parents=True,exist_ok=True)
    sample,audit=load_samples(zip_path,cfg['max_rows_per_class'],cfg['chunk_size'],seed)
    sample=normalize_columns(sample)
    filenames=sample['_capture_file'].astype(str)
    mask=filenames.str.lower().str.contains(holdout_contains.lower(),regex=False)
    if not mask.any():
        raise ValueError(f'No sampled records matched {holdout_contains!r}; choose a filename substring')
    labels=sample['Label'].astype(str).str.strip().str.upper().ne('BENIGN').astype(int)
    features=[c for c in sample if c.lower() not in DROP and not c.startswith('_')]
    x=sample[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Exact duplicate removal globally, before partitioning.
    keep=~x.assign(_label=labels.to_numpy()).duplicated()
    x=x.loc[keep].reset_index(drop=True);labels=labels.loc[keep].reset_index(drop=True);mask=mask.loc[keep].reset_index(drop=True)
    xtr=x.loc[~mask]; ytr=labels.loc[~mask];xt=x.loc[mask];yt=labels.loc[mask]
    if ytr.nunique()!=2: raise ValueError('Training portion must have both classes')
    ix_train,ix_val=train_test_split(np.arange(len(ytr)),test_size=.15,stratify=ytr,random_state=seed)
    result={'held_out_file_pattern':holdout_contains,'note':'File-held-out test; attacks exclusive to held-out file may be unseen classes; metrics do not imply unseen-family detection','train_rows':len(ix_train),'validation_rows':len(ix_val),'test_rows':len(xt),'test_positive_fraction':float(yt.mean()),'models':{}}
    for name in models:
        try:
            model=pipeline(name,seed);model.fit(xtr.iloc[ix_train],ytr.iloc[ix_train]);
            val=model.predict_proba(xtr.iloc[ix_val])[:,1]
            test=model.predict_proba(xt)[:,1]
            result['models'][name]={'validation':metrics(ytr.iloc[ix_val],val),'test':metrics(yt,test)}
        except (ImportError,ModuleNotFoundError,OSError) as e:
            result['models'][name]={'skipped':str(e)}
    (out/'day_holdout.json').write_text(json.dumps(result,indent=2))
    return result
