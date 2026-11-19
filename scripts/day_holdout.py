import argparse,json
from cybersentinel.day_holdout import run_day_holdout
p=argparse.ArgumentParser()
p.add_argument('--zip',default='data/MachineLearningCSV.zip')
p.add_argument('--holdout',default='Friday-WorkingHours-Afternoon-DDos')
p.add_argument('--models',nargs='+',default=['mlp','random_forest'])
a=p.parse_args()
print(json.dumps(run_day_holdout(a.zip,holdout_contains=a.holdout,models=a.models),indent=2))
