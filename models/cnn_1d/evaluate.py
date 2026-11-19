"""RF/XGBoost-compatible window metrics and per-run summaries."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix
from .data import dump

def metrics(frame):
    t=frame.true.to_numpy();p=frame.predicted.to_numpy();both=len(np.unique(t))==2
    return dict(evaluated_runs=sorted(frame.run_id.unique()),skipped_runs=[],windows=len(frame),classes_in_evaluation=sorted(int(x) for x in np.unique(t)),
        precision=float(precision_score(t,p,zero_division=0)) if both else None,
        recall=float(recall_score(t,p,zero_division=0)) if 1 in t else None,
        f1=float(f1_score(t,p,zero_division=0)) if both else None,
        balanced_accuracy=float(balanced_accuracy_score(t,p)) if both else None,
        confusion_matrix=confusion_matrix(t,p,labels=[0,1]).tolist(),
        per_run=frame.groupby("run_id").agg(true=("true","first"),fraction_predicted_failing=("predicted","mean"),mean_failing_score=("failing_score","mean")).reset_index().to_dict("records"))

def score(model,dataset,device,batch_size,threshold):
    model.eval();scores=[];losses=[]
    with torch.no_grad():
        for x,t,y,w in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            logits=model(x.to(device),t.to(device));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y.to(device),reduction="none")
            losses.extend(loss.cpu().tolist());scores.extend(torch.sigmoid(logits).cpu().tolist())
    f=dataset.frame.drop(columns=["local_index","array_index"]).copy();f["failing_score"]=scores;f["predicted"]=(np.array(scores)>=threshold).astype(int)
    f["loss"]=losses
    return f,float(f.groupby("run_id").loss.mean().mean())

def report(frame,path,protocol,threshold,dataset):
    path.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path/(protocol+"_predictions.csv"),index=False)
    result=dict(protocol=protocol,dataset=dataset,threshold=threshold,targets={"nozzle_health":{"trained":True,"evaluation":metrics(frame)}})
    if "loss" in frame:result["run_balanced_loss"]=float(frame.groupby("run_id").loss.mean().mean())
    dump(path/"metrics.json",result);return result
