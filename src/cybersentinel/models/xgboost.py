def build(seed=42):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.08,tree_method='hist',n_jobs=2,random_state=seed,eval_metric='logloss')
