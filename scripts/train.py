import argparse
from cybersentinel.training import train
p=argparse.ArgumentParser();p.add_argument('--out',default='artifacts');p.add_argument('--models',nargs='*');a=p.parse_args()
for result in train(a.out,models=a.models): print(result['model'], 'F1=',round(result['f1'],4),'PR-AUC=',round(result['pr_auc'],4))
