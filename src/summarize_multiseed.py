import json, os, numpy as np
files=[r'G:\Datasets\HyperCross\runs\leaf_20261005_214059\routeA_history.json',r'G:\Datasets\HyperCross\runs\leaf_20261005_221026\routeA_history.json',r'G:\Datasets\HyperCross\runs\leaf_20261005_221133\routeA_history.json']
rows=[]
for f in files:
    h=json.load(open(f)); b=max(h,key=lambda x:x['r2_mean']); rows.append({'run':os.path.dirname(f),'epoch':b['epoch'],'r2_mean':b['r2_mean'],'r2':b['r2']})
vals=[r['r2_mean'] for r in rows]; out={'runs':rows,'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1))}; print(json.dumps(out,indent=2)); open(r'G:\Datasets\HyperCross\leaf_multiseed.json','w').write(json.dumps(out,indent=2))
