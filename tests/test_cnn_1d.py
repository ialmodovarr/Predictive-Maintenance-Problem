"""Focused checks for the hybrid model input and training contracts."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.cnn_1d.model import HybridCNN
from models.cnn_1d.data import resample,normalization,Windows,FEATURES
from models.cnn_1d.evaluate import metrics

class CNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg=dict(hz=767,coverage=.99,min_span=9.99,max_gap=.0025)
        self.t=np.arange(-1,7672)/767+.0001
        self.x=np.stack([self.t,self.t*2,self.t*3],axis=1)
        self.seg=np.zeros(len(self.t))
    def test_shapes_gradients_and_serialization(self):
        m=HybridCNN();self.assertEqual(sum(p.numel() for p in m.parameters()),17185)
        x=torch.randn(2,3,7670);t=torch.randn(2,13,requires_grad=True)
        self.assertEqual(m.deep(x).shape,(2,64));z=m(x,t);self.assertEqual(z.shape,(2,))
        torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.])).backward()
        self.assertIsNotNone(t.grad);self.assertIsNotNone(m.deep[0].weight.grad)
        m.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"weights.pt";torch.save(m.state_dict(),path);other=HybridCNN();other.load_state_dict(torch.load(path,weights_only=True));other.eval()
            torch.testing.assert_close(m(x,t),other(x,t))
    def test_interpolation(self):
        out=resample(self.t,self.x,self.seg,0,10,self.cfg)
        np.testing.assert_allclose(out[0],np.arange(7670)/767,atol=1e-6)
        self.assertEqual(out.shape,(3,7670))
    def test_boundary_rejection(self):
        with self.assertRaisesRegex(ValueError,"unsupported_boundary"):
            resample(self.t[1:],self.x[1:],self.seg[1:],0,10,self.cfg)
    def test_gap_and_segment_rejection(self):
        mask=np.ones(len(self.t),bool);mask[100:104]=False
        with self.assertRaisesRegex(ValueError,"gap"):resample(self.t[mask],self.x[mask],self.seg[mask],0,10,self.cfg)
        self.seg[100:]=1
        with self.assertRaisesRegex(ValueError,"segment"):resample(self.t,self.x,self.seg,0,10,self.cfg)
    def test_invalid_xyz(self):
        self.x[100,1]=np.nan
        with self.assertRaisesRegex(ValueError,"invalid_acceleration"):resample(self.t,self.x,self.seg,0,10,self.cfg)
    def test_run_balanced_normalization_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,(n,value) in enumerate([(2,0.),(6,10.)]):
                p=Path(tmp)/str(i);p.mkdir();paths.append(p)
                np.save(p/"acceleration.npy",np.full((n,3,10),value,dtype="float32"));np.save(p/"auxiliary.npy",np.full((n,13),value,dtype="float32"))
                pd.DataFrame(dict(run_id=[str(i)]*n,window_id=list(range(n)),true=[i]*n)).to_csv(p/"windows.csv",index=False)
            norm=normalization(paths);np.testing.assert_allclose(norm["acceleration"]["mean"],5);np.testing.assert_allclose(norm["auxiliary"]["scale"],5)
            ds=Windows(paths,norm);self.assertAlmostEqual(ds.weights[:2].sum(),ds.weights[2:].sum())
            losses=np.array([2.,2.,4.,4.,4.,4.,4.,4.]);self.assertAlmostEqual(float(np.mean(losses*ds.weights)),3.,places=6)
            ds.close()
    def test_metric_convention(self):
        f=pd.DataFrame(dict(run_id=['a','a','b','b'],true=[0,0,1,1],predicted=[0,1,0,1],failing_score=[.1,.7,.2,.8]))
        r=metrics(f);self.assertEqual(r['confusion_matrix'],[[1,1],[1,1]]);self.assertEqual(r['f1'],.5)
    def test_training_artifacts_and_resume(self):
        from unittest.mock import patch
        from models.cnn_1d.train import fit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache={}
            for i in range(4):
                name=f"run_{i:04d}";p=root/name;p.mkdir();cache[name]=p
                np.save(p/"acceleration.npy",np.random.default_rng(i).normal(size=(2,3,80)).astype("float32"))
                np.save(p/"auxiliary.npy",np.full((2,13),i,dtype="float32"))
                pd.DataFrame(dict(run_id=[name]*2,window_id=['w1','w2'],true=[i%2]*2)).to_csv(p/"windows.csv",index=False)
                (p/"manifest.json").write_text(json.dumps(dict(run=name)))
            cfg=dict(seed=42,threads=2,dropout=.2,lr=.001,weight_decay=.0001,min_delta=.0001,epochs=2,batch_size=2,threshold=.5,patience=10,preprocessing={})
            def prep(dataset,name,config):return cache[name]
            with patch('models.cnn_1d.train.prepare',side_effect=prep),patch('models.cnn_1d.train.labels',return_value=[0,1]):
                exp=fit('mock',list(cache)[:2],list(cache)[2:],[],cfg,root/'experiment')
                report=json.loads((exp/'artifacts/metrics.json').read_text());self.assertEqual(report['protocol'],'validation')
                self.assertEqual(report['targets']['nozzle_health']['evaluation']['windows'],4)
                self.assertTrue((exp/'artifacts/resume_checkpoint.pt').exists())
                fit('mock',list(cache)[:2],list(cache)[2:],[],cfg,exp,resume=True)
    def test_stratified_folds_are_disjoint(self):
        from unittest.mock import patch
        from models.cnn_1d.train import folds
        names=[f'run_{i:04d}' for i in range(24)]
        with patch('models.cnn_1d.train.labels',return_value=[0]*12+[1]*12):
            fs=folds('mock',names,42)
        validation=[]
        for tr,va in fs:
            self.assertEqual(len(tr),18);self.assertEqual(len(va),6);self.assertFalse(set(tr)&set(va));validation+=va
        self.assertEqual(sorted(validation),names)
    def test_scheduler_patience(self):
        p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([p],lr=.001)
        s=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,threshold_mode='abs',threshold=.0001)
        for _ in range(4):s.step(1.)
        self.assertEqual(opt.param_groups[0]['lr'],.001);s.step(1.);self.assertEqual(opt.param_groups[0]['lr'],.0005)
if __name__=='__main__':unittest.main()
