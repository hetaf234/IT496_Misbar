#!/usr/bin/env python3
"""Fine-tune the Food-101 DINOv2 checkpoint on immutable final_food_dataset_v2.

Only train/ and val/ are opened.  The test directory is deliberately rejected.
"""
from __future__ import annotations

import argparse, copy, csv, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXPECTED = {"classes": 121, "train": 76198, "val": 3630}

class DinoClassifier(nn.Module):
    def __init__(self, backbone, outputs):
        super().__init__(); self.backbone = backbone
        self.head = nn.Linear(int(getattr(backbone, "embed_dim", 768)), outputs)
    def forward(self, x): return self.head(self.backbone(x))

class FolderDataset(Dataset):
    def __init__(self, root, mapping, transform):
        self.root, self.mapping, self.transform = Path(root), mapping, transform
        self.samples = []
        for name, idx in mapping.items():
            self.samples += [(p, idx) for p in sorted((self.root / name).iterdir()) if p.is_file()]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        with Image.open(p) as im: return self.transform(im.convert("RGB")), y

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); temp.replace(path)

def save_evaluation_artifacts(output, classes, confusion, recall):
    with (output / "best_confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["true_class", *classes])
        for label, row in zip(classes, confusion.tolist()): writer.writerow([label, *row])
    with (output / "best_per_class_recall.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["class_index", "class_name", "validation_recall"])
        for index, (label, value) in enumerate(zip(classes, recall.tolist())): writer.writerow([index, label, f"{value:.8f}"])

def transforms_for(cfg):
    train = transforms.Compose([transforms.RandomResizedCrop(cfg["image_size"], scale=(.70, 1.), ratio=(.80, 1.25), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 7, interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.RandomErasing(.05, scale=(.02,.10))])
    val = transforms.Compose([transforms.Resize(cfg["eval_resize"], interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(cfg["image_size"]), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, val

def metrics(logits, targets, confusion):
    pred = logits.argmax(1); confusion.index_put_((targets.cpu(), pred.cpu()), torch.ones_like(pred.cpu(), dtype=torch.int64), accumulate=True)
    top5 = (logits.topk(5, 1).indices == targets[:, None]).any(1).sum().item()
    return pred.eq(targets).sum().item(), top5

def evaluate(model, loader, device, classes):
    model.eval(); c = torch.zeros(len(classes), len(classes), dtype=torch.int64); total=correct=top5=0; loss_sum=0.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for x,y in loader:
            x,y=x.to(device, non_blocking=True),y.to(device, non_blocking=True); z=model(x); n=len(y)
            loss_sum += F.cross_entropy(z,y).item()*n; a,b=metrics(z,y,c); correct+=a; top5+=b; total+=n
    recall = c.diag().float()/c.sum(1).clamp_min(1); precision=c.diag().float()/c.sum(0).clamp_min(1); f1=(2*precision*recall/(precision+recall).clamp_min(1e-12))
    return {"loss":loss_sum/total,"top1_accuracy":correct/total,"top5_accuracy":top5/total,"macro_f1":f1.mean().item(),"count":total}, c, recall

def score(m): return (round(m["top1_accuracy"], 6), m["macro_f1"], -m["loss"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--torch-home",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); root=a.dataset
    if (root/"test").exists() is False: raise RuntimeError("Missing immutable test directory")
    old=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if old.get("model_name") != "DINOv2_ViTB14" or len(old.get("classes",[])) != 101: raise RuntimeError("Expected compatible 101-class DINOv2 ViT-B/14 checkpoint")
    # ImageFolder supplies the exact lexical class_to_idx ordering used by the loaders.
    indexer=datasets.ImageFolder(root/"train"); new_map=indexer.class_to_idx
    if len(indexer)!=EXPECTED["train"] or len(new_map)!=EXPECTED["classes"]: raise RuntimeError("Unexpected train split")
    if {p.name for p in (root/"val").iterdir() if p.is_dir()} != set(new_map): raise RuntimeError("Train/validation mappings differ")
    old_map={name:i for i,name in enumerate(old["classes"])}; transferred=sorted(set(old_map)&set(new_map)); missing=sorted(set(old_map)-set(new_map))
    if len(transferred)!=101 or missing: raise RuntimeError(f"Food-101 class mismatch: {missing}")
    counts=torch.bincount(torch.tensor(indexer.targets),minlength=len(new_map)).float(); weights=len(indexer)/(len(new_map)*counts)
    if len(weights)!=121 or torch.any(counts==0): raise RuntimeError("Invalid class weights")
    torch.hub.set_dir(str(a.torch_home)); backbone=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",pretrained=False,trust_repo=True)
    model=DinoClassifier(backbone,101); model.load_state_dict(old["model_state_dict"],strict=True)
    new_head=nn.Linear(model.head.in_features,121).to(model.head.weight.dtype)
    with torch.no_grad():
        for name in transferred:
            new_head.weight[new_map[name]].copy_(model.head.weight[old_map[name]]); new_head.bias[new_map[name]].copy_(model.head.bias[old_map[name]])
    model.head=new_head; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("CUDA GPU is required")
    model.to(device); train_tf,val_tf=transforms_for(cfg); train=FolderDataset(root/"train",new_map,train_tf); val=FolderDataset(root/"val",new_map,val_tf)
    if len(train)!=EXPECTED["train"] or len(val)!=EXPECTED["val"] or any(sum(1 for p in (root/"val"/n).iterdir() if p.is_file())!=30 for n in new_map): raise RuntimeError("Dataset split check failed")
    bs=next(b for b in cfg["batch_candidates"] if b<=36); train_loader=DataLoader(train,batch_size=bs,shuffle=True,num_workers=cfg["num_workers"],pin_memory=True,persistent_workers=True); val_loader=DataLoader(val,batch_size=max(bs,32),shuffle=False,num_workers=cfg["num_workers"],pin_memory=True,persistent_workers=True)
    # Required pre-launch forward/loss check; the weighted criterion is training-only.
    x,y=next(iter(train_loader)); x,y=x.to(device),y.to(device); criterion=nn.CrossEntropyLoss(weight=weights.to(device),label_smoothing=cfg["label_smoothing"]); z=model(x); sanity_loss=criterion(z,y); sanity_loss.backward(); model.zero_grad(set_to_none=True)
    verification={"previous_classes_transferred":len(transferred),"newly_initialized_classes":121-len(transferred),"unmatched_previous_classes":missing,"classifier_outputs":model.head.out_features,"class_weight_entries":len(weights),"weights_training_only":True,"test_accessed":False,"sanity_forward_and_loss":float(sanity_loss.detach()),"class_to_idx":new_map}
    atomic_json(a.output/"preflight.json",verification); print(json.dumps(verification,sort_keys=True),flush=True)
    resume=None; resume_path=a.output/"last_checkpoint.pt"
    if a.resume:
        if not resume_path.exists(): raise FileNotFoundError(f"--resume requested but {resume_path} does not exist")
        resume=torch.load(resume_path,map_location="cpu",weights_only=False)
        if resume.get("class_to_idx") != new_map: raise RuntimeError("Cannot resume: class_to_idx mismatch")
        model.load_state_dict(resume["model_state_dict"],strict=True); print(f"RESUME_REQUESTED epoch={resume.get('epoch')}",flush=True)
    scaler=torch.amp.GradScaler("cuda"); best={"top1_accuracy":-1.,"macro_f1":-1.,"loss":float("inf")}; ema=None; stale=0; epoch0=0 if not resume else resume["epoch"]
    for stage,epochs in (("full",cfg["full_epochs"]),) if resume else (("head",cfg["head_epochs"]),("full",cfg["full_epochs"])):
        for q in model.backbone.parameters(): q.requires_grad=(stage=="full")
        for q in model.head.parameters(): q.requires_grad=True
        groups=[{"params":model.head.parameters(),"lr":cfg["head_lr"] if stage=="head" else cfg["full_head_lr"]}]
        if stage=="full": groups.insert(0,{"params":model.backbone.parameters(),"lr":cfg["backbone_lr"]}); ema=copy.deepcopy(model).eval()
        opt=torch.optim.AdamW(groups,weight_decay=cfg["weight_decay"]); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs*len(train_loader)))
        if resume:
            opt.load_state_dict(resume["optimizer_state_dict"]); sched.load_state_dict(resume["scheduler_state_dict"]); print("RESUME_OPTIMIZER_AND_SCHEDULER_LOADED",flush=True)
        for _ in range(epochs):
            epoch0+=1; model.train(); total=loss_sum=0.
            for batch_index,(x,y) in enumerate(train_loader):
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16): loss=criterion(model(x),y)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); sched.step(); total+=len(y); loss_sum+=loss.item()*len(y)
                if batch_index==0: print(f"FIRST_TRAIN_BATCH_COMPLETED epoch={epoch0} stage={stage} batch_size={len(y)}",flush=True)
                if ema:
                    for e,v in zip(ema.parameters(),model.parameters()): e.data.mul_(cfg["ema_decay"]).add_(v.data,alpha=1-cfg["ema_decay"])
            chosen=ema if ema else model; val_m,conf,recall=evaluate(chosen,val_loader,device,indexer.classes); row={"epoch":epoch0,"stage":stage,"train_loss":loss_sum/total,"validation":val_m,"lr_backbone":cfg["backbone_lr"] if stage=="full" else 0.,"lr_head":groups[-1]["lr"]}; atomic_json(a.output/"last_metrics.json",row)
            payload={"model_state_dict":chosen.state_dict(),"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":sched.state_dict(),"epoch":epoch0,"stage":stage,"metrics":val_m,"class_to_idx":new_map,"classes":indexer.classes,"architecture":"DINOv2_ViTB14","preprocessing":{"image_size":336,"eval_resize":378,"mean":MEAN,"std":STD},"config":cfg,"transfer":verification}
            torch.save(payload,a.output/"last_checkpoint.pt")
            if score(val_m)>score(best): best=val_m; stale=0; torch.save(payload,a.output/"best_checkpoint.pt"); torch.save(conf,a.output/"best_confusion_matrix.pt"); save_evaluation_artifacts(a.output,indexer.classes,conf,recall); atomic_json(a.output/"best_metrics.json",row)
            else: stale+= stage=="full"
            print(json.dumps({"epoch":epoch0,"stage":stage,"train_loss":row["train_loss"],**val_m,"lr":row["lr_head"]}),flush=True)
            if stage=="full" and stale>=cfg["early_stopping_patience"]: return
if __name__=="__main__": main()
