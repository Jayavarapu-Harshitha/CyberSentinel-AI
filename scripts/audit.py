import argparse,json
from cybersentinel.data import run_audit
p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',default='artifacts');args=p.parse_args()
print(json.dumps(run_audit(args.zip,args.out),indent=2))
