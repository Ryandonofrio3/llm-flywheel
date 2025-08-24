#!/usr/bin/env python3
import os, time, argparse, logging
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
CIFAR5=["apple","mushroom","orange","pear","sweet_pepper"]
class DirDataset(Dataset):
    def __init__(self, root):
        self.items=[]; 
        for c in CIFAR5:
            d=os.path.join(root,c)
            if not os.path.isdir(d): raise FileNotFoundError(d)
            self.items += [(os.path.join(d,f), c) for f in os.listdir(d) if f.lower().endswith(".jpg")]
        self.cls={c:i for i,c in enumerate(CIFAR5)}
        self.tf=transforms.Compose([transforms.Resize((224,224)),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,c=self.items[i]; x=Image.open(p).convert("RGB")
        return self.tf(x), self.cls[c]
def load_student(ckpt:str, device:str):
    m=models.mobilenet_v3_small(weights=None)
    m.classifier[3]=nn.Linear(m.classifier[3].in_features,len(CIFAR5))
    sd=torch.load(ckpt, map_location=device); m.load_state_dict(sd); m.to(device).eval()
    return m
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cifar5_run/images/test")
    ap.add_argument("--ckpt", default="cifar5_run/artifacts/mobilenet.pt")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count()//2))
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    device="mps" if torch.backends.mps.is_available() else "cpu"
    ds=DirDataset(args.data_dir); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=False)
    m=load_student(args.ckpt, device)
    # warmup
    xb,_=next(iter(dl)); xb=xb.to(device); 
    with torch.no_grad(): [m(xb) for _ in range(5)]
    n=0; correct=0; t0=time.time()
    with torch.no_grad():
        for xb,yb in dl:
            xb=xb.to(device)
            out=m(xb).argmax(1).cpu()
            correct += (out==yb).sum().item(); n+=xb.size(0)
    dt=time.time()-t0
    ips=n/dt
    logging.info(f"throughput {ips:.1f} img/s  latency {1000/ips:.3f} ms/img  acc={correct/n:.3f}  device={device}  threads={args.threads}")
if __name__=="__main__": main()
