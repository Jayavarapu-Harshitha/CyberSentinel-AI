import numpy as np
from .evaluation import metrics

def blend(p_mlp,p_tree,weight):
    if not 0<=weight<=1: raise ValueError('weight must be between 0 and 1')
    return weight*np.asarray(p_mlp)+(1-weight)*np.asarray(p_tree)

def select_weight(y_val,p_mlp,p_tree):
    # Validation-only optimization, never test-set tuning.
    trials=[(float(w),metrics(y_val,blend(p_mlp,p_tree,w))['f1']) for w in np.linspace(0,1,21)]
    return sorted(trials,key=lambda t:(-t[1],abs(t[0]-.5)))[0][0]
