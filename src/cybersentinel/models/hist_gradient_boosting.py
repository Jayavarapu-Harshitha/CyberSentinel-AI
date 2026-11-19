from sklearn.ensemble import HistGradientBoostingClassifier
def build(seed=42):
    return HistGradientBoostingClassifier(max_iter=150,learning_rate=0.08,max_leaf_nodes=31,random_state=seed)
