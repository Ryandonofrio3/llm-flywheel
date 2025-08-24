#!/usr/bin/env python3
# cifar5_llm_distill.py
import os, io, sys, csv, json, time, base64, random, logging, argparse
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

import requests
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv
load_dotenv()

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cifar5")

# ---------------- constants ----------------
CIFAR5 = ["apple","mushroom","orange","pear","sweet_pepper"]
LLM_MODEL_DEFAULT = "google/gemini-2.5-flash"

# ---------------- utils ----------------
def set_seed(seed:int=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def normalize_llm_label(x:str) -> Optional[str]:
    if not x: return None
    y = x.strip().lower().replace("-", "_").replace(" ", "_")
    # plural to singular
    y = {"apples":"apple","mushrooms":"mushroom","oranges":"orange","pears":"pear",
         "sweet_peppers":"sweet_pepper","sweet_pepper":"sweet_pepper"}.get(y, y)
    # common variants
    if y in {"sweet_pepper","sweet__pepper","sweetpepper"}: y="sweet_pepper"
    return y if y in CIFAR5 else None

# ---------------- data prep ----------------
def build_cifar5(out_dir:str, sharpen:bool=True, max_per_class:int=None) -> Tuple[List[str], List[str]]:
    """
    Downloads CIFAR-100, filters the 5 classes, upscales to 224, saves to disk.
    Returns (train_paths, test_paths).
    """
    try:
        ensure_dir(out_dir)
        tr_img_dir = os.path.join(out_dir, "images", "train")
        te_img_dir = os.path.join(out_dir, "images", "test")
        ensure_dir(tr_img_dir); ensure_dir(te_img_dir)

        # torchvision datasets
        tr = tvds.CIFAR100(root=os.path.join(out_dir,"_torchvision"), train=True, download=True)
        te = tvds.CIFAR100(root=os.path.join(out_dir,"_torchvision"), train=False, download=True)

        name_list = list(tr.classes)  # fine label names
        sel_idx = [name_list.index(c) for c in CIFAR5]

        # upscale + optional unsharp
        def process_and_save(split_name:str, ds, out_base:str) -> List[str]:
            per_class_count = defaultdict(int)
            saved_paths=[]
            for i, (img, y) in enumerate(zip(ds.data, ds.targets)):
                if y not in sel_idx: continue
                cls = name_list[y]
                if max_per_class and per_class_count[cls] >= max_per_class: continue
                im = Image.fromarray(img).convert("RGB").resize((224,224), Image.BICUBIC)
                if sharpen:
                    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))
                cls_dir = os.path.join(out_base, cls); ensure_dir(cls_dir)
                p = os.path.join(cls_dir, f"{cls}_{per_class_count[cls]:05d}.jpg")
                if not os.path.exists(p):
                    im.save(p, quality=90, optimize=True)
                saved_paths.append(p)
                per_class_count[cls]+=1
            log.info(f"{split_name}: {sum(per_class_count.values())} saved; dist={dict(per_class_count)}")
            return saved_paths

        tr_paths = process_and_save("train", tr, tr_img_dir)
        te_paths = process_and_save("test",  te, te_img_dir)
        return tr_paths, te_paths
    except Exception as e:
        log.exception(f"build_cifar5 failed: {e}")
        raise

# ---------------- LLM labeling ----------------
def openrouter_label_image(img_path:str, model:str, api_key:str, labels:List[str], timeout:int=40, max_retries:int=5) -> Optional[Dict]:
    """
    Calls OpenRouter chat completions with an image and strict JSON response.
    Returns {'label':<str>, 'confidence':float} or None.
    Enhanced with exponential backoff.
    """
    prompt = (
        "Classify this 224x224 photo into one of:\n"
        '["apple","mushroom","orange","pear","sweet_pepper"].\n'
        "If fruit vs vegetable is ambiguous, prefer the fruit.\n"
        'Return JSON only: {"label":"<one>","confidence":0-1,"why":"<=12 words"}.'
    )
    b64 = base64.b64encode(open(img_path,"rb").read()).decode()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 100,
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":"You are a precise image labeler. Respond with strict JSON only."},
            {"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff with jitter
            if attempt > 0:
                delay = min(300, (2 ** attempt) + random.uniform(0, 1))
                time.sleep(delay)
                
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]["content"]
            data = json.loads(msg)
            lab = normalize_llm_label(data.get("label",""))
            conf = float(data.get("confidence", 0))
            if lab in labels and 0 <= conf <= 1:
                return {"label": lab, "confidence": conf, "raw_response": data}
            else:
                raise ValueError(f"bad json fields: {data}")
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:  # Rate limit
                log.warning(f"Rate limited on {os.path.basename(img_path)}, attempt {attempt+1}")
                continue
            elif r.status_code >= 500:  # Server error
                log.warning(f"Server error {r.status_code} on {os.path.basename(img_path)}, attempt {attempt+1}")
                continue
            else:
                log.warning(f"HTTP error {r.status_code} on {os.path.basename(img_path)}: {e}")
                return None
        except Exception as e:
            if attempt < max_retries-1:
                log.warning(f"Error on {os.path.basename(img_path)}, attempt {attempt+1}: {e}")
                continue
            log.warning(f"Final failure on {os.path.basename(img_path)}: {e}")
            return None
    return None

def parallel_dual_pass_label(img_path:str, model:str, api_key:str, labels:List[str], conf_thresh:float) -> Optional[Dict]:
    """
    Does dual pass labeling with parallel calls for speed.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(openrouter_label_image, img_path, model, api_key, labels)
        future_b = executor.submit(openrouter_label_image, img_path, model, api_key, labels)
        
        a = future_a.result()
        b = future_b.result()
        
    if not a or not b: 
        return None
    if a["label"] != b["label"]: 
        return None
        
    conf = 0.5*(a["confidence"] + b["confidence"])
    if conf < conf_thresh: 
        return None
        
    return {
        "label": a["label"], 
        "confidence": conf,
        "raw_responses": [a.get("raw_response"), b.get("raw_response")]
    }

def load_progress(progress_file: str) -> Dict:
    """Load progress from JSON file if it exists."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": {}, "failed": set(), "last_saved": 0}

def save_progress(progress_file: str, progress: Dict):
    """Save progress to JSON file."""
    # Convert set to list for JSON serialization
    progress_copy = progress.copy()
    progress_copy["failed"] = list(progress_copy["failed"])
    with open(progress_file, 'w') as f:
        json.dump(progress_copy, f, indent=2)

def process_image_batch(img_paths: List[str], model: str, api_key: str, conf_thresh: float, 
                       progress: Dict, progress_file: str, lock: Lock) -> List[Tuple]:
    """Process a batch of images with a single worker."""
    results = []
    for img_path in img_paths:
        # Check if already processed
        with lock:
            if img_path in progress["completed"] or img_path in progress["failed"]:
                continue
                
        try:
            r = parallel_dual_pass_label(img_path, model, api_key, CIFAR5, conf_thresh)
            with lock:
                if r:
                    progress["completed"][img_path] = {
                        "label": r["label"], 
                        "confidence": r["confidence"],
                        "raw_responses": r.get("raw_responses", [])
                    }
                    results.append((img_path, r["label"], r["confidence"]))
                else:
                    progress["failed"].add(img_path)
                    
                # Save progress every 10 items
                if len(progress["completed"]) % 10 == 0 and len(progress["completed"]) > progress["last_saved"]:
                    save_progress(progress_file, progress)
                    progress["last_saved"] = len(progress["completed"])
                    
        except Exception as e:
            log.warning(f"Error processing {os.path.basename(img_path)}: {e}")
            with lock:
                progress["failed"].add(img_path)
                
    return results

def label_trainset(out_dir: str, max_items: int, model: str, api_key: str, conf_thresh: float = 0.0, 
                  max_workers: int = 10) -> str:
    """
    LLM-labels up to max_items from train images using parallel processing.
    Saves progress and can resume from interruptions.
    Returns path to review_labels.csv.
    """
    train_dir = os.path.join(out_dir, "images", "train")
    if not os.path.isdir(train_dir): 
        raise RuntimeError("train images not found; run --step prep or --step all first")
        
    # Get all image paths
    all_paths = []
    for c in CIFAR5:
        cdir = os.path.join(train_dir, c)
        if os.path.isdir(cdir):
            all_paths += [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(".jpg")]
    
    random.shuffle(all_paths)
    if max_items: 
        all_paths = all_paths[:max_items]
    
    progress_file = os.path.join(out_dir, "labeling_progress.json")
    detailed_file = os.path.join(out_dir, "labeling_detailed.jsonl")
    
    # Load existing progress
    progress = load_progress(progress_file)
    progress["failed"] = set(progress["failed"])  # Convert back to set
    
    # Filter out already processed images
    remaining_paths = [p for p in all_paths if p not in progress["completed"] and p not in progress["failed"]]
    
    log.info(f"LLM labeling: total={len(all_paths)}, completed={len(progress['completed'])}, "
             f"failed={len(progress['failed'])}, remaining={len(remaining_paths)} with {model}")
    
    if not remaining_paths:
        log.info("All images already processed!")
    else:
        # Split paths into batches for workers
        batch_size = max(1, len(remaining_paths) // max_workers)
        batches = [remaining_paths[i:i + batch_size] for i in range(0, len(remaining_paths), batch_size)]
        
        lock = Lock()
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(process_image_batch, batch, model, api_key, conf_thresh, 
                                       progress, progress_file, lock)
                futures.append(future)
            
            # Monitor progress
            completed_count = len(progress["completed"])
            while any(not f.done() for f in futures):
                time.sleep(5)
                with lock:
                    current_count = len(progress["completed"])
                    if current_count > completed_count:
                        log.info(f"Progress: {current_count}/{len(all_paths)} completed, "
                                f"{len(progress['failed'])} failed")
                        completed_count = current_count
            
            # Collect results
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    log.error(f"Batch processing error: {e}")
    
    # Save final progress
    save_progress(progress_file, progress)
    
    # Write detailed results to JSONL
    with open(detailed_file, 'w') as f:
        for img_path, data in progress["completed"].items():
            record = {
                "path": img_path,
                "label": data["label"],
                "confidence": data["confidence"],
                "raw_responses": data.get("raw_responses", [])
            }
            f.write(json.dumps(record) + '\n')
    
    # Write CSV files
    ok = [(p, data["label"], data["confidence"]) for p, data in progress["completed"].items()]
    
    labels_csv = os.path.join(out_dir, "labels_llm.csv")
    with open(labels_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "llm_label", "confidence"])
        for p, y, c in ok:
            w.writerow([p, y, round(c, 4)])
    
    review_csv = os.path.join(out_dir, "review_labels.csv")
    with open(review_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "llm_label", "confidence", "label_final"])
        for p, y, c in ok:
            w.writerow([p, y, round(c, 4), y])  # copy llm_label to label_final for easy edits
    
    log.info(f"Labeling complete: {len(ok)} successful, {len(progress['failed'])} failed")
    log.info(f"Wrote {labels_csv}, {review_csv}, {detailed_file}")
    log.info(f"Progress saved to {progress_file}")
    
    return review_csv

# ---------------- dataset for training ----------------
class CsvImageDataset(Dataset):
    def __init__(self, csv_path:str, class_names:List[str], augment:bool):
        self.rows=[]
        with open(csv_path) as f:
            r=csv.DictReader(f)
            for row in r:
                y=row.get("label_final") or row.get("llm_label")
                y=normalize_llm_label(y or "")
                if y in class_names:
                    self.rows.append({"path":row["path"], "label":y})
        if not self.rows: raise RuntimeError(f"no rows in {csv_path}")
        self.class_to_idx={c:i for i,c in enumerate(class_names)}
        ops=[transforms.Resize((224,224))]
        if augment:
            ops=[transforms.Resize((224,224)),
                 transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
                 transforms.RandomRotation(5),
                 transforms.ColorJitter(0.15,0.15,0.15,0.02)]
        self.tf = transforms.Compose(ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.rows)
    def __getitem__(self, i:int):
        p=self.rows[i]["path"]; y=self.class_to_idx[self.rows[i]["label"]]
        x=Image.open(p).convert("RGB"); return self.tf(x), y

class DirImageDataset(Dataset):
    def __init__(self, root:str, class_names:List[str]):
        self.items=[]
        for c in class_names:
            cdir=os.path.join(root,c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith(".jpg"): self.items.append((os.path.join(cdir,fn), c))
        self.class_to_idx={c:i for i,c in enumerate(class_names)}
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, i:int):
        p,c=self.items[i]; y=self.class_to_idx[c]
        x=Image.open(p).convert("RGB"); return self.tf(x), y

# ---------------- training ----------------
def train_mobilenet(train_csv:str, test_dir:str, out_dir:str, epochs:int=30, batch_size:int=128, lr:float=3e-4):
    try:
        ensure_dir(out_dir)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pin = False

        dtr = CsvImageDataset(train_csv, CIFAR5, augment=True)
        dva = DirImageDataset(test_dir, CIFAR5)

        dltr=DataLoader(dtr,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=pin)
        dlva=DataLoader(dva,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=pin)

        model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(CIFAR5))
        model=model.to(device)

        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.05)
        opt=optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best=0.0; best_path=os.path.join(out_dir,"mobilenet.pt")

        for ep in range(1, epochs+1):
            # train
            model.train(); n=c=los=0.0
            for xb,yb in dltr:
                xb,yb=xb.to(device), yb.to(device)
                opt.zero_grad(); out=model(xb); L=loss_fn(out,yb)
                L.backward(); opt.step()
                los+=L.item()*xb.size(0); c+=(out.argmax(1)==yb).sum().item(); n+=xb.size(0)
            tr_acc=c/n; tr_loss=los/n

            # eval
            model.eval(); n=c=0.0; ys=[]; ps=[]
            with torch.no_grad():
                for xb,yb in dlva:
                    xb,yb=xb.to(device), yb.to(device)
                    out=model(xb); pr=out.argmax(1)
                    c+=(pr==yb).sum().item(); n+=xb.size(0)
                    ys+=yb.cpu().tolist(); ps+=pr.cpu().tolist()
            val_acc=c/n
            if val_acc>best:
                best=val_acc
                torch.save(model.state_dict(), best_path)
            sch.step()
            log.info(f"ep{ep}: train_acc={tr_acc:.3f} loss={tr_loss:.3f} val_acc={val_acc:.3f}")

        rpt=classification_report(ys,ps,target_names=CIFAR5, digits=4, zero_division=0)
        cm=confusion_matrix(ys,ps).tolist()
        with open(os.path.join(out_dir,"eval.json"),"w") as f:
            json.dump({"val_acc":best,"report":rpt,"confusion_matrix":cm,"classes":CIFAR5}, f)
        log.info(f"best val_acc={best:.3f}  saved={best_path}")
    except Exception as e:
        log.exception(f"train failed: {e}")
        raise

# ---------------- CLI ----------------
def main():
    ap=argparse.ArgumentParser(description="CIFAR5: LLM labels → MobileNet student")
    ap.add_argument("--out_dir", default="cifar5_run")
    ap.add_argument("--step", choices=["prep","label","train","all"], default="all")
    ap.add_argument("--max_per_class", type=int, default=None, help="cap saved images per class")
    ap.add_argument("--max_label", type=int, default=1500, help="max images to LLM-label from train")
    ap.add_argument("--llm_model", default=LLM_MODEL_DEFAULT)
    ap.add_argument("--conf_thresh", type=float, default=0.0, help="dual-pass mean confidence gate")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_workers", type=int, default=10, help="max parallel workers for LLM calls")
    args=ap.parse_args()

    set_seed(args.seed)
    try:
        if args.step in {"prep","all"}:
            log.info("STEP prep: build CIFAR5 images (224x224)")
            tr_paths, te_paths = build_cifar5(args.out_dir, sharpen=True, max_per_class=args.max_per_class)
            log.info(f"prep done: train={len(tr_paths)} test={len(te_paths)}")

        if args.step in {"label","all"}:
            key=os.getenv("OPENROUTER_API_KEY","")
            if not key:
                log.error("OPENROUTER_API_KEY not set; export and rerun.")
                sys.exit(2)
            log.info("STEP label: LLM labeling train images")
            review_csv = label_trainset(args.out_dir, args.max_label, args.llm_model, key, args.conf_thresh, args.max_workers)
            log.info(f"label done. edit if desired: {review_csv}")

        if args.step in {"train","all"}:
            log.info("STEP train: train MobileNet on reviewed labels; eval on test subset")
            review_csv = os.path.join(args.out_dir,"review_labels.csv")
            if not os.path.isfile(review_csv):
                alt = os.path.join(args.out_dir,"labels_llm.csv")
                if not os.path.isfile(alt):
                    raise RuntimeError("no labels found; run --step label first")
                # build a default review file
                with open(alt) as fin, open(review_csv,"w",newline="") as fout:
                    r=csv.DictReader(fin); w=csv.writer(fout)
                    w.writerow(["path","llm_label","confidence","label_final"])
                    for row in r: w.writerow([row["path"], row["llm_label"], row["confidence"], row["llm_label"]])
                log.info(f"created {review_csv} from labels_llm.csv")
            test_dir = os.path.join(args.out_dir,"images","test")
            art_dir = os.path.join(args.out_dir,"artifacts")
            train_mobilenet(review_csv, test_dir, art_dir, epochs=args.epochs, batch_size=args.batch_size)
            log.info("train done.")
    except Exception as e:
        log.exception(f"fatal: {e}")
        sys.exit(1)

if __name__=="__main__":
    main()


