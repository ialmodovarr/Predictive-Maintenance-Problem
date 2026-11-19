"""Inference uses saved preprocessing without refitting it."""
import json
from pathlib import Path
import torch
from .data import prepare,Windows
from .model import HybridCNN
from .evaluate import score,report

def predict(experiment,dataset,names,evaluation=False):
    exp=Path(experiment);art=exp/"artifacts";cfg=json.loads((art/"config.json").read_text());norm=json.loads((art/"preprocessing.json").read_text())
    if evaluation:
        provenance=json.loads((art/"provenance.json").read_text())
        if provenance["dataset"]!=dataset:raise ValueError("Evaluation dataset differs from training")
        if provenance.get("smoke_window_limit"):raise ValueError("Smoke models cannot receive final-test evaluation")
        split=json.loads((art/"splits.json").read_text());names=split["test"]
        if split["validation"]:raise ValueError("Use a final-trained model for reserved test evaluation")
    torch.set_num_threads(cfg["threads"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=HybridCNN(cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(art/"model.pt",map_location=device,weights_only=True))
    ds=Windows([prepare(dataset,n,cfg["preprocessing"]) for n in names],norm);frame,_=score(model,ds,device,cfg["batch_size"],cfg["threshold"])
    if evaluation:report(frame,exp/"test_evaluation","test",cfg["threshold"],dataset)
    else:
        dest=exp/"predictions";dest.mkdir(exist_ok=True)
        for name,f in frame.groupby("run_id"):
            f=f.drop(columns=["loss"]);f["nozzle_health_failing_score"]=f.pop("failing_score");f["nozzle_health_predicted"]=f.pop("predicted").map({0:"healthy",1:"failing"});f.to_csv(dest/(name+"_predictions.csv"),index=False)
    ds.close()
    return frame
