from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
root=Path('results'); rows=[]
for p in root.rglob('official_metrics.json'):
    try: rows.append(json.loads(p.read_text(encoding='utf-8')))
    except Exception as e: print('skip',p,e)
if not rows: raise SystemExit('No official_metrics.json files found')
df=pd.DataFrame(rows).sort_values(['dataset','attack','seed','defense']); out=Path('artifacts/stage1_aggregate'); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'benchmark_runs.csv',index=False)
summary=df.groupby(['dataset','attack','defense'],dropna=False).agg(asr_mean=('asr','mean'),asr_std=('asr','std'),ca_mean=('clean_accuracy','mean'),ca_std=('clean_accuracy','std'),runs=('seed','count')).reset_index(); summary.to_csv(out/'benchmark_group_stats.csv',index=False)
lines=['# GSDD-Bench Stage-1 Summary','',f'- Runs: `{len(df)}`','', '| Dataset | Attack | Defense | Runs | ASR mean | CA mean |','|---|---|---|---:|---:|---:|']
for _,r in summary.iterrows(): lines.append(f"| {r.dataset} | {r.attack} | {r.defense} | {int(r.runs)} | {r.asr_mean:.4f} | {r.ca_mean:.4f} |")
(out/'STAGE1_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
print(out)
