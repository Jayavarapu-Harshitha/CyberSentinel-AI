import numpy as np,pandas as pd
from cybersentinel.data import prepare
from cybersentinel.evaluation import metrics
from cybersentinel.fusion import blend,select_weight
from cybersentinel.training import pipeline

def test_metrics():
    r=metrics([0,0,1,1],[.1,.9,.8,.2]);assert r['tp']==1 and r['fp']==1 and r['fn']==1 and r['tn']==1

def test_blend():
    assert np.allclose(blend([.2,.8],[.6,.4],.5),[.4,.6])
    assert 0<=select_weight([0,1,0,1],[.1,.9,.2,.8],[.4,.6,.3,.7])<=1

def test_prepare_no_leakage_identifiers():
    df=pd.DataFrame({'Label':['BENIGN']*20+['DDoS']*20,'Flow Duration':range(40),'Flow ID':['id']*40,'Destination Port':[80]*40})
    splits,info=prepare(df)
    assert 'Flow ID' not in info['feature_columns']
    assert 'Destination Port' not in info['feature_columns']
    assert sum(len(x) for x,y in splits.values())==40

def test_cpu_mlp():
    rng=np.random.default_rng(4);x=pd.DataFrame({'a':rng.normal(size=60),'b':rng.normal(size=60)})
    y=pd.Series([0]*30+[1]*30)
    model=pipeline('mlp',42);model.fit(x,y)
    assert model.predict_proba(x.iloc[:2]).shape==(2,2)
