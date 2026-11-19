"""Hybrid CNN CLI; all generated outputs are separate from original recordings."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.cnn_1d.data import prepare
from models.cnn_1d.train import partitions,folds,fit,cross_validate
from models.cnn_1d.predict import predict

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["prepare","train","cross-validate","final-train","evaluate","predict"])
    p.add_argument("--dataset",required=True,choices=["mock","real"]);p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"));p.add_argument("--split-file")
    p.add_argument("--runs",nargs="+");p.add_argument("--experiment");p.add_argument("--cv-experiment");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true")
    a=p.parse_args();cfg=json.loads(a.config.read_text());dev,test=partitions(a.dataset,a.split_file)
    base=Path(__file__).parent/"experiments"
    if a.command=="prepare":
        for name in a.runs or dev:
            if name not in dev:raise ValueError("Preparation command is development-only; test preparation occurs during final evaluation")
            prepare(a.dataset,name,cfg["preprocessing"])
        return
    if not a.experiment or Path(a.experiment).name!=a.experiment:raise ValueError("Provide a simple unique experiment name")
    out=base/a.experiment
    if a.command in ("evaluate","predict"):
        if a.command=="predict" and not a.runs:raise ValueError("Prediction requires --runs")
        predict(out,a.dataset,a.runs,a.command=="evaluate");return
    limit=8 if a.smoke else None
    if a.smoke:cfg["epochs"]=2
    if a.command=="train":
        tr,va=folds(a.dataset,dev,cfg["seed"])[0]
        if a.smoke:
            from models.cnn_1d.data import labels
            tr=[next(n for n,y in zip(tr,labels(a.dataset,tr)) if y==c) for c in (0,1)]
            va=[next(n for n,y in zip(va,labels(a.dataset,va)) if y==c) for c in (0,1)]
        fit(a.dataset,tr,va,test,cfg,out,limit,resume=a.resume)
    elif a.command=="cross-validate":cross_validate(a.dataset,dev,test,cfg,out,limit)
    else:
        if not a.cv_experiment or Path(a.cv_experiment).name!=a.cv_experiment:raise ValueError("Provide --cv-experiment")
        plan=json.loads((base/a.cv_experiment/"artifacts/final_plan.json").read_text())
        if plan["smoke"] or plan["development"]!=dev or plan["test"]!=test or plan["dataset"]!=a.dataset or plan["config"]!=cfg:raise ValueError("Final plan does not match this dataset/config or comes from smoke CV")
        cfg["epochs"]=plan["epochs"];fit(a.dataset,dev,[],test,cfg,out,final_schedule=plan["learning_rates"])
if __name__=="__main__":main()
