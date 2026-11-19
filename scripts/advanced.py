import argparse,json
from pathlib import Path
from cybersentinel.advanced import anomaly_experiment,feature_explanation,ablation_experiment,latency_experiment
from cybersentinel.config import ROOT
p=argparse.ArgumentParser()
p.add_argument('experiment',choices=['anomaly','explain','ablation','latency','all'])
p.add_argument('--out',default=str(ROOT/'artifacts'))
a=p.parse_args()
funcs={'anomaly':anomaly_experiment,'explain':feature_explanation,'ablation':ablation_experiment,'latency':latency_experiment}
for name,func in funcs.items():
    if a.experiment in (name,'all'):
        print(name,json.dumps(func(Path(a.out)),indent=2)[:1800])
