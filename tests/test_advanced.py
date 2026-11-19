import numpy as np
import pytest
from cybersentinel.fusion import blend,select_weight
from cybersentinel.evaluation import metrics
from cybersentinel.models.registry import build

def test_all_five_model_modules_resolve():
    for name in ('logistic_regression','random_forest','mlp'):
        assert hasattr(build(name,42),'fit')
    # Optional XGBoost and LightGBM modules are implemented but their native wheels
    # are not required for the base test environment.
    import importlib
    for name in ('xgboost','lightgbm'):
        assert callable(importlib.import_module('cybersentinel.models.'+name).build)

def test_fusion_bounds_and_weight_selection():
    y=np.array([0,0,1,1]);a=np.array([.1,.2,.9,.8]);b=np.array([.2,.1,.8,.9])
    assert np.allclose(blend(a,b,.5),(a+b)/2)
    assert 0 <= select_weight(y,a,b) <= 1
    with pytest.raises(ValueError): blend(a,b,1.2)

def test_confusion_metrics():
    m=metrics([0,0,1,1],[.1,.7,.8,.2]);assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,1,1)
    assert m['precision']==.5 and m['recall']==.5
