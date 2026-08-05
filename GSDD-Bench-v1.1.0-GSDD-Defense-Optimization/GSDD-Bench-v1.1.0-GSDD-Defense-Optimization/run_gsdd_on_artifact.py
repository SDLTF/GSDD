from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from scipy.stats import rankdata

from gsdd_core.graph_ops import build_normalized_adjacency, build_normalized_laplacian, node_degree
from gsdd_core.models import SupervisedGCN, DGIModel
from gsdd_core.train import train_supervised, train_dgi, extract_supervised_hidden, extract_dgi_hidden
from gsdd_core.spectral import band_energies, log_band_gain, decompose_delta_gain
from gsdd_core.artifact import normalize_attack_bundle


def fpr95(y,s):
    fpr,tpr,_=roc_curve(y,s); idx=np.flatnonzero(tpr>=.95); return float(fpr[idx[0]]) if len(idx) else 1.0

def rank01(x): return (rankdata(x,method='average')-.5)/len(x)

def metrics(y,s):
    order=np.argsort(s)[::-1]
    out={'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)),'fpr_at_95_tpr':fpr95(y,s)}
    for frac in (.01,.05):
        k=max(1,int(math.ceil(len(y)*frac))); out[f'recall_at_top_{int(frac*100)}pct']=float(y[order[:k]].sum()/max(1,y.sum()))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=1027); ap.add_argument('--hidden-dim',type=int,default=32); ap.add_argument('--bands',type=int,default=5)
    ap.add_argument('--supervised-epochs',type=int,default=200); ap.add_argument('--ssl-epochs',type=int,default=200); ap.add_argument('--patience',type=int,default=40)
    ap.add_argument('--filter-fraction',type=float,default=.01); ap.add_argument('--lr',type=float,default=.01); ap.add_argument('--weight-decay',type=float,default=5e-4)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA is required; CPU fallback is disabled')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); device=torch.device('cuda')
    art=Path(args.artifact); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    b=normalize_attack_bundle(torch.load(art/'artifact.pt',map_location='cpu',weights_only=False))
    x=b['poison_x'].float().to(device); y=b['poison_y'].long().to(device); edge=b['poison_train_edge_index'].long().to(device)
    n=x.shape[0]; train_idx=b['poison_train_idx'].long().to(device); val_idx=b['val_idx'].long().to(device); attach=b['attach_idx'].long().cpu().numpy()
    if y.shape[0] != n:
        raise RuntimeError(f'Artifact normalization failed: poison_x has {n} nodes but poison_y has {y.shape[0]} labels')
    train_mask=torch.zeros(n,dtype=torch.bool,device=device); train_mask[train_idx]=True
    val_mask=torch.zeros(n,dtype=torch.bool,device=device); val_mask[val_idx]=True
    adj=build_normalized_adjacency(edge,n,device); lap=build_normalized_laplacian(edge,n,device)
    sup=SupervisedGCN(x.shape[1],args.hidden_dim,int(y.max().item()+1),.5).to(device)
    ssl=DGIModel(x.shape[1],args.hidden_dim).to(device)
    t0=time.time(); hs=train_supervised(sup,x,y,adj,train_mask,val_mask,args.supervised_epochs,args.lr,args.weight_decay,args.patience,False)
    hd=train_dgi(ssl,x,adj,args.ssl_epochs,args.lr,args.weight_decay,args.patience,False)
    _,sup_hidden=extract_supervised_hidden(sup,x,adj); ssl_hidden=extract_dgi_hidden(ssl,x,adj)
    input_raw,_=band_energies(lap,x,args.bands,1e-9)
    components=[]; rows={}
    for li,(sh,dh) in enumerate(zip(sup_hidden,ssl_hidden),start=1):
        sr,sd=band_energies(lap,sh,args.bands,1e-9); dr,dd=band_energies(lap,dh,args.bands,1e-9)
        sg=log_band_gain(sr,input_raw,sh.shape[1],x.shape[1],1e-9); dg=log_band_gain(dr,input_raw,dh.shape[1],x.shape[1],1e-9)
        delta=sg-dg; _,shape,_=decompose_delta_gain(delta)
        raw=torch.linalg.vector_norm(delta,dim=1).detach().cpu().numpy(); shp=torch.linalg.vector_norm(shape,dim=1).detach().cpu().numpy(); dist=torch.linalg.vector_norm(sd-dd,dim=1).detach().cpu().numpy()
        rows[f'raw_l{li}']=raw; rows[f'shape_l{li}']=shp; rows[f'distribution_l{li}']=dist; components.extend([rank01(raw),rank01(shp),rank01(dist)])
    hybrid=np.stack(components,axis=1).mean(axis=1); rows['spectral_hybrid']=hybrid
    candidates=np.unique(b['poison_train_idx'].long().cpu().numpy()); truth=np.isin(candidates,attach).astype(np.int64)
    if truth.sum()==0 or truth.sum()==len(truth): raise SystemExit('Detection candidates must contain both clean and poisoned nodes')
    score_table={'node_id':candidates,'is_poison':truth}
    report={}
    for name,full in rows.items():
        sc=full[candidates]; score_table[name]=sc; report[name]=metrics(truth,sc)
    pd.DataFrame(score_table).to_csv(out/'node_scores.csv',index=False)
    # Operational filter: remove the highest-ranked fixed fraction, without using poison labels.
    filter_count=max(1,int(math.ceil(args.filter_fraction*len(candidates)))); ranked=candidates[np.argsort(hybrid[candidates])[::-1]]; removed=ranked[:filter_count]
    filtered=np.setdiff1d(candidates,removed,assume_unique=False); torch.save(torch.as_tensor(filtered,dtype=torch.long),out/'filtered_train_idx.pt')
    summary={'artifact':str(art),'seed':args.seed,'device':torch.cuda.get_device_name(0),'candidate_count':int(len(candidates)),'poison_count':int(truth.sum()),'filter_fraction':args.filter_fraction,'filter_count':int(filter_count),'supervised_best_epoch':hs.best_epoch,'ssl_best_epoch':hd.best_epoch,'seconds':time.time()-t0,'metrics':report}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# GSDD Official-artifact Detection Summary','',f"- Artifact: `{art}`",f"- CUDA device: `{summary['device']}`",f"- Candidates: `{len(candidates)}`",f"- Known poisoned nodes (evaluation only): `{truth.sum()}`",f"- Filtered nodes: `{filter_count}`",'', '| Score | AUROC | AUPRC | FPR@95TPR | Recall@Top1% | Recall@Top5% |','|---|---:|---:|---:|---:|---:|']
    for name,m in report.items(): lines.append(f"| {name} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['fpr_at_95_tpr']:.4f} | {m['recall_at_top_1pct']:.4f} | {m['recall_at_top_5pct']:.4f} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(out/'filtered_train_idx.pt')
if __name__=='__main__': main()
