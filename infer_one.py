#!/usr/bin/env python3
import os, argparse, json, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models

def load_student(artifacts:str, device:str):
    d=json.load(open(os.path.join(artifacts,"eval.json"))); classes=d["classes"]
    m=models.mobilenet_v3_small(weights=None)
    m.classifier[3]=nn.Linear(m.classifier[3].in_features, len(classes))
    sd=torch.load(os.path.join(artifacts,"mobilenet.pt"), map_location=device)
    m.load_state_dict(sd); m.eval().to(device); return m, classes

TF=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="cifar5_run/artifacts")
    ap.add_argument("--image", required=True)
    args=ap.parse_args()
    device="mps" if torch.backends.mps.is_available() else "cpu"
    m, classes = load_student(args.artifacts, device)
    x=TF(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p=torch.softmax(m(x), dim=1).squeeze(0).cpu()
    idx=int(p.argmax().item()); print(json.dumps({"pred":classes[idx], "probs":{c:round(float(p[i]),4) for i,c in enumerate(classes)}}))

if __name__=="__main__": main()
