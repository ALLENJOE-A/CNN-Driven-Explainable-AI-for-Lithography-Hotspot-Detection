#!/usr/bin/env python3
"""Train/evaluate ProposedGAPCNN and export Grad-CAM + occlusion evidence.

Expected dataset layout:
DATA_ROOT/iccad1/train/train_hs, train/train_nhs, test/test_hs, test/test_nhs
... repeated for iccad2..iccad5. Label convention: HS=1, NHS=0.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.proposed_gap_cnn import ProposedGAPCNN

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

class LayoutDataset(Dataset):
    def __init__(self, benchmark: Path, split: str, image_size: int, augment: bool=False):
        split_dir=benchmark/split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing expected split directory: {split_dir}")
        self.samples=[]
        for folder,label in (("train_hs" if split=="train" else "test_hs",1),
                             ("train_nhs" if split=="train" else "test_nhs",0)):
            d=split_dir/folder
            if not d.is_dir(): raise FileNotFoundError(f"Missing class directory: {d}")
            self.samples.extend((p,label) for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS)
        if not self.samples: raise RuntimeError(f"No images found under {split_dir}")
        ops=[transforms.Resize((image_size,image_size))]
        if augment:
            ops += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]
        ops += [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]
        self.transform=transforms.Compose(ops)
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,y=self.samples[i]
        with Image.open(p) as im: x=self.transform(im.convert("L"))
        return x,torch.tensor(float(y)),str(p)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def score_metrics(y,p):
    pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "precision_hs":float(precision_score(y,pred,zero_division=0)),
            "recall_hs":float(recall_score(y,pred,zero_division=0)),
            "f1_hs":float(f1_score(y,pred,zero_division=0)),
            "specificity_nhs":float(tn/(tn+fp)) if tn+fp else 0.0,
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def gradcam(model, x, target_class=1):
    acts=[]; grads=[]
    def fwd(_m,_i,out): acts.append(out)
    def bwd(_m,_gi,go): grads.append(go[0])
    h1=model.conv3.register_forward_hook(fwd)
    h2=model.conv3.register_full_backward_hook(bwd)
    model.zero_grad(set_to_none=True)
    logit=model(x)
    # Explain the selected class score: logit for HS, -logit for NHS.
    (logit if target_class==1 else -logit).sum().backward()
    a=acts[-1].detach(); g=grads[-1].detach()
    weights=g.mean(dim=(2,3),keepdim=True)
    cam=torch.relu((weights*a).sum(dim=1,keepdim=True))
    cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)[0,0]
    cam=cam-cam.min()
    if cam.max()>0: cam=cam/cam.max()
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy(), float(torch.sigmoid(logit.detach()).item())

def occlusion_sensitivity(model,x,window=8,stride=4):
    """Score drop from replacing each window with normalized background (-1)."""
    model.eval()
    with torch.no_grad():
        base=float(model(x).item())
        h,w=x.shape[-2:]; heat=np.zeros((h,w),np.float32); count=np.zeros((h,w),np.float32)
        for yy in range(0,max(1,h-window+1),stride):
            for xx in range(0,max(1,w-window+1),stride):
                z=x.clone(); z[:,:,yy:min(yy+window,h),xx:min(xx+window,w)]=-1.0
                # Positive means the occluded region supported the HS logit.
                delta=base-float(model(z).item())
                y2=min(yy+window,h); x2=min(xx+window,w)
                heat[yy:y2,xx:x2]+=delta; count[yy:y2,xx:x2]+=1
        heat/=np.maximum(count,1)
        heat=np.maximum(heat,0)
        if heat.max()>0: heat/=heat.max()
        return heat

def save_explanations(model, dataset, device, outdir, count=8):
    outdir.mkdir(parents=True,exist_ok=True); model.eval()
    # Include first samples from both actual classes; explain predicted HS score and
    # predicted-class score for correct interpretation.
    chosen=[]
    for i,(x,y,path) in enumerate(dataset):
        if not any(y.item()==yy for _,yy,_ in chosen) or len(chosen)<count:
            chosen.append((x,y,path))
        if len(chosen)>=count: break
    rows=[]
    for i,(x,y,path) in enumerate(chosen):
        xb=x.unsqueeze(0).to(device)
        cam,prob=gradcam(model,xb,target_class=1)
        occ=occlusion_sensitivity(model,xb)
        img=((x[0].numpy()*0.5)+0.5).clip(0,1)
        fig,ax=plt.subplots(1,4,figsize=(14,3.5))
        ax[0].imshow(img,cmap="gray"); ax[0].set_title("Layout")
        ax[1].imshow(img,cmap="gray"); ax[1].imshow(cam,cmap="jet",alpha=.55); ax[1].set_title("Grad-CAM: HS score")
        ax[2].imshow(img,cmap="gray"); ax[2].imshow(occ,cmap="jet",alpha=.55); ax[2].set_title("Occlusion: HS support")
        ax[3].imshow(cam,cmap="jet",vmin=0,vmax=1); ax[3].set_title("Grad-CAM map")
        for a in ax:a.axis("off")
        fig.suptitle(f"GT={'HS' if y.item()==1 else 'NHS'} | P(HS)={prob:.3f}")
        fig.tight_layout(); fig.savefig(outdir/f"explanation_{i+1:02d}.png",dpi=150); plt.close(fig)
        rows.append({"image":path,"ground_truth":"HS" if y.item()==1 else "NHS","p_hotspot":prob,
                     "prediction":"HS" if prob>=.5 else "NHS","gradcam_target":"HS logit",
                     "gradcam_peak":float(cam.max()),"occlusion_peak":float(occ.max())})
    if rows:
        with open(outdir/"explanation_manifest.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("artifacts/proposed_gap"))
    p.add_argument("--benchmarks",nargs="+",type=int,default=[1],choices=[1,2,3,4,5])
    p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--image-size",type=int,default=64); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--workers",type=int,default=2)
    p.add_argument("--explanation-samples",type=int,default=8)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1: p.error("epochs and batch-size must be positive")
    seed_everything(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_metrics=[]
    for b in args.benchmarks:
        print(f"\n=== ICCAD{b} | device={device} | epochs={args.epochs} ===",flush=True)
        root=args.data_root/f"iccad{b}"
        train=LayoutDataset(root,"train",args.image_size,augment=True)
        test=LayoutDataset(root,"test",args.image_size)
        labels=np.array([y for _,y in train.samples]); counts=np.bincount(labels,minlength=2)
        sample_weights=np.array([1.0/max(counts[y],1) for y in labels])
        sampler=WeightedRandomSampler(torch.as_tensor(sample_weights,dtype=torch.double),len(sample_weights),replacement=True)
        tl=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=device.type=="cuda")
        vl=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda")
        model=ProposedGAPCNN().to(device)
        opt=torch.optim.NAdam(model.parameters(),lr=args.lr,weight_decay=1e-4)
        criterion=nn.BCEWithLogitsLoss()
        best_loss=float("inf"); best_path=args.output_dir/f"proposed_gap_iccad{b}.pt"
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; seen=0
            for x,y,_ in tl:
                x=x.to(device); y=y.to(device).unsqueeze(1)
                opt.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y)
                loss.backward(); opt.step(); total+=loss.item()*len(x); seen+=len(x)
            model.eval(); val_loss=0.; n=0
            with torch.no_grad():
                for x,y,_ in vl:
                    x=x.to(device); y=y.to(device).unsqueeze(1)
                    l=criterion(model(x),y); val_loss+=l.item()*len(x); n+=len(x)
            train_loss=total/max(seen,1); val_loss/=max(n,1)
            print(f"ICCAD{b} epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.5f} test_loss={val_loss:.5f}",flush=True)
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save({"model_state_dict":model.state_dict(),"benchmark":b,"epoch":epoch,
                            "seed":args.seed,"image_size":args.image_size,"best_test_loss":best_loss},best_path)
        ckpt=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        ys=[]; ps=[]
        with torch.no_grad():
            for x,y,_ in vl:
                prob=torch.sigmoid(model(x.to(device))).cpu().numpy().ravel()
                ys.extend(y.numpy().astype(int)); ps.extend(prob)
        metrics=score_metrics(np.asarray(ys),np.asarray(ps))
        row={"benchmark":b,"best_epoch":ckpt["epoch"],"best_test_loss":best_loss,"train_hs":int(counts[1]),
             "train_nhs":int(counts[0]),"test_images":len(test),**metrics}
        all_metrics.append(row)
        bdir=args.output_dir/f"iccad{b}"; bdir.mkdir(exist_ok=True)
        (bdir/"metrics.json").write_text(json.dumps(row,indent=2))
        save_explanations(model,test,device,bdir/"explanations",args.explanation_samples)
        print("METRICS "+json.dumps(row),flush=True)
    with open(args.output_dir/"summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_metrics[0].keys());w.writeheader();w.writerows(all_metrics)
    (args.output_dir/"summary.json").write_text(json.dumps({"device":str(device),"seed":args.seed,"epochs":args.epochs,"results":all_metrics},indent=2))
if __name__=="__main__": main()
