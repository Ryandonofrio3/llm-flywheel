#!/usr/bin/env python3
import os, argparse, json, logging, torch, torch.nn as nn
from torchvision import models
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export")

def load_student(artifacts:str, device:str):
    d=json.load(open(os.path.join(artifacts,"eval.json")))
    classes=d["classes"]; n=len(classes)
    m=models.mobilenet_v3_small(weights=None)
    m.classifier[3]=nn.Linear(m.classifier[3].in_features, n)
    sd=torch.load(os.path.join(artifacts,"mobilenet.pt"), map_location=device)
    m.load_state_dict(sd); m.eval().to(device)
    return m, classes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="cifar5_run/artifacts")
    ap.add_argument("--out_dir", default="cifar5_run/exports")
    ap.add_argument("--fp16", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    m, classes = load_student(args.artifacts, device)
    ex=torch.randn(1,3,224,224, device=device)
    if args.fp16 and device!="cpu":
        m.half(); ex=ex.half()

    # TorchScript
    traced=torch.jit.trace(m, ex)
    ts_path=os.path.join(args.out_dir,"mobilenet_cifar5.ts")
    traced.save(ts_path)
    log.info(f"saved TorchScript: {ts_path}")

    # ONNX
    onnx_path=os.path.join(args.out_dir,"mobilenet_cifar5.onnx")
    torch.onnx.export(
        m, ex, onnx_path, input_names=["input"], output_names=["logits"],
        opset_version=17, dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}
    )
    log.info(f"saved ONNX: {onnx_path}")
    with open(os.path.join(args.out_dir,"labels.json"),"w") as f:
        json.dump(classes,f)
    log.info("saved labels.json")

if __name__=="__main__": main()
