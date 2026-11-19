"""Run-isolated training, cross-validation, and development-only final fitting."""
import json,random,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from .data import ROOT,FEATURES,dump,labels,prepare,normalization,Windows,eligibility_summary
from .model import HybridCNN
from .evaluate import score,report

def partitions(dataset,split_file=None):
    path=Path(split_file) if split_file else ROOT/"data/mock_metadata/split.json"
    if dataset=="real" and split_file is None:raise ValueError("Real data requires an explicit split JSON")
    s=json.loads(path.read_text());dev=sorted(s["train_validation"]);test=sorted(s["test"])
    if len(set(dev+test))!=len(dev+test):raise ValueError("Duplicate/overlapping split runs")
    labels(dataset,dev+test)
    return dev,test

def folds(dataset,dev,seed):
    y=labels(dataset,dev)
    if min(y.count(0),y.count(1))<4:raise ValueError("Four-fold CV requires at least four runs of each class")
    return [([dev[i] for i in a],[dev[i] for i in b]) for a,b in StratifiedKFold(4,shuffle=True,random_state=seed).split(dev,y)]

def fit(dataset,train_names,val_names,test_names,cfg,out,limit=None,final_schedule=None,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume);art=out/"artifacts";art.mkdir(exist_ok=resume)
    seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(cfg["threads"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if set(train_names)&set(val_names+test_names) or set(val_names)&set(test_names):raise ValueError("Partition overlap")
    if set(labels(dataset,train_names))!={0,1}:raise ValueError("Training requires both classes")
    caches=[prepare(dataset,n,cfg["preprocessing"]) for n in train_names]
    if all((p/"eligibility.csv").exists() for p in caches):dump(art/"training_eligibility.json",eligibility_summary(caches))
    norm=normalization(caches,limit);tr=Windows(caches,norm,limit)
    va=Windows([prepare(dataset,n,cfg["preprocessing"]) for n in val_names],norm,limit) if val_names else None
    if resume:
        old_provenance=json.loads((art/"provenance.json").read_text())
        if old_provenance["dataset"]!=dataset or old_provenance.get("smoke_window_limit")!=limit or old_provenance["cache_manifests"]!=[json.loads((p/"manifest.json").read_text()) for p in caches]:
            raise ValueError("Resume source provenance differs")
        previous=json.loads((art/"config.json").read_text());saved=json.loads((art/"splits.json").read_text())
        if previous!=cfg or saved!=dict(train=train_names,validation=val_names,test=test_names) or json.loads((art/"preprocessing.json").read_text())!=norm:
            raise ValueError("Resume inputs/config differ from the saved experiment")
    dump(art/"preprocessing.json",norm);dump(art/"config.json",cfg);dump(art/"splits.json",dict(train=train_names,validation=val_names,test=test_names))
    dump(art/"provenance.json",dict(dataset=dataset,features=FEATURES,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=str(device),smoke_window_limit=limit,cache_manifests=[json.loads((p/"manifest.json").read_text()) for p in caches]))
    model=HybridCNN(cfg["dropout"]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"],betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=3,threshold=cfg["min_delta"],threshold_mode="abs",min_lr=1e-6)
    history=[];best=float("inf");significant=float("inf");bad=0;best_epoch=0
    start_epoch=1
    if resume:
        state=torch.load(art/"resume_checkpoint.pt",map_location=device,weights_only=False)
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        start_epoch=state["epoch"]+1;best_epoch=state["best_epoch"];best=state["best_loss"];significant=state["significant"];bad=state["bad_epochs"]
        torch.set_rng_state(state["torch_rng"].cpu());random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(state["cuda_rng"])
        history=pd.read_csv(art/"history.csv").replace({np.nan:None}).to_dict("records")
        if bad>=cfg["patience"]:start_epoch=cfg["epochs"]+1
    for epoch in range(start_epoch,cfg["epochs"]+1):
        if final_schedule is not None:
            for g in opt.param_groups:g["lr"]=final_schedule[epoch-1]
        lr=opt.param_groups[0]["lr"];model.train();total=0.
        for x,t,y,w in DataLoader(tr,batch_size=cfg["batch_size"],shuffle=True):
            opt.zero_grad();z=model(x.to(device),t.to(device));loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y.to(device),reduction="none")
            weighted=loss*w.to(device);weighted.mean().backward();opt.step();total+=weighted.detach().sum().item()
        val=None
        if va is not None:
            _,val=score(model,va,device,cfg["batch_size"],cfg["threshold"])
            if val<best:best=val;best_epoch=epoch;torch.save(model.state_dict(),art/"model.pt")
            if val<significant-cfg["min_delta"]:significant=val;bad=0
            else:bad+=1
            scheduler.step(val)
        else:best_epoch=epoch;torch.save(model.state_dict(),art/"model.pt")
        history.append(dict(epoch=epoch,train_loss=total/len(tr),validation_loss=val,learning_rate=lr));pd.DataFrame(history).to_csv(art/"history.csv",index=False)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best_epoch=best_epoch,best_loss=best,significant=significant,bad_epochs=bad,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),art/"resume_checkpoint.pt")
        print(f"epoch {epoch}: train={total/len(tr):.6f}, validation={val}",flush=True)
        if va is not None and bad>=cfg["patience"]:break
    dump(art/"selection.json",dict(best_epoch=best_epoch,final_schedule=final_schedule))
    if va is not None:
        model.load_state_dict(torch.load(art/"model.pt",map_location=device,weights_only=True));f,_=score(model,va,device,cfg["batch_size"],cfg["threshold"]);report(f,art,"validation",cfg["threshold"],dataset)
    tr.close()
    if va is not None:va.close()
    return out

def cross_validate(dataset,dev,test,cfg,out,limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);frames=[];epochs=[];histories=[]
    for i,(tr,va) in enumerate(folds(dataset,dev,cfg["seed"])):
        fold=fit(dataset,tr,va,test,cfg,out/f"fold_{i}",limit)
        a=fold/"artifacts";frames.append(pd.read_csv(a/"validation_predictions.csv"));epochs.append(json.loads((a/"selection.json").read_text())["best_epoch"]);histories.append(pd.read_csv(a/"history.csv").learning_rate.tolist())
    n=max(1,int(np.median(epochs)));schedule=[float(np.median([h[min(i,len(h)-1)] for h in histories])) for i in range(n)]
    report(pd.concat(frames,ignore_index=True),out/"artifacts","nozzle_health_oof",cfg["threshold"],dataset)
    dump(out/"artifacts/final_plan.json",dict(epochs=n,learning_rates=schedule,development=dev,test=test,dataset=dataset,config=cfg,smoke=limit is not None))
