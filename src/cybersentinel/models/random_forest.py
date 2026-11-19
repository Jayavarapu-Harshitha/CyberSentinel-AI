from sklearn.ensemble import RandomForestClassifier
def build(seed=42):
    return RandomForestClassifier(n_estimators=120,max_depth=18,min_samples_leaf=2,n_jobs=2,random_state=seed,class_weight='balanced_subsample')
