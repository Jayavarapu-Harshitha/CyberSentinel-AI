"""Optional local SHAP explanations for XGBoost/LightGBM tree pipelines."""
from pathlib import Path
import joblib,numpy as np,pandas as pd

def shap_explain(out,model_name,frame,max_rows=20):
    if model_name not in ('xgboost','lightgbm','random_forest'):
        raise ValueError('SHAP explanation supported only for tree estimators')
    try: import shap
    except ImportError as e: raise RuntimeError('Install optional dependency: pip install shap') from e
    model=joblib.load(Path(out)/f'{model_name}.joblib')
    frame=frame.iloc[:max_rows].copy()
    transformed=model[:-1].transform(frame)
    estimator=model.named_steps['model']
    explainer=shap.TreeExplainer(estimator)
    values=explainer.shap_values(transformed)
    if isinstance(values,list): values=values[-1]
    values=np.asarray(values)
    if values.ndim==3: values=values[:,:,1]
    if values.ndim!=2: raise RuntimeError(f'Unexpected SHAP shape: {values.shape}')
    return pd.DataFrame(values,columns=frame.columns,index=frame.index)
