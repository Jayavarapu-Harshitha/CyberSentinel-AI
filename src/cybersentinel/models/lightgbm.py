def build(seed=42):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=150,learning_rate=0.08,num_leaves=31,n_jobs=2,random_state=seed,verbosity=-1)
