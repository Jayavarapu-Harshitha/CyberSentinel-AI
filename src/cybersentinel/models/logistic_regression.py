from sklearn.linear_model import LogisticRegression
def build(seed=42):
    return LogisticRegression(max_iter=500, random_state=seed, class_weight='balanced')
