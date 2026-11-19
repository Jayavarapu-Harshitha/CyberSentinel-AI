"""Optional local SHAP export for one of the trained tree classifiers."""
import argparse
from pathlib import Path
import pandas as pd
from cybersentinel.explain import shap_explain
p=argparse.ArgumentParser()
p.add_argument('--model',choices=['xgboost','lightgbm','random_forest'],default='random_forest')
p.add_argument('--out',default='artifacts')
p.add_argument('--rows',type=int,default=20)
a=p.parse_args()
out=Path(a.out)
val=pd.read_pickle(out/'val.pkl').drop(columns='_target')
values=shap_explain(out,a.model,val,max_rows=a.rows)
path=out/f'shap_{a.model}.csv'
values.to_csv(path,index=False)
print('Saved',path)
