#!/usr/bin/env python3
import os, sys, json, argparse, logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cm")

def plot_cm(cm, labels, title, out_png):
    cm = np.array(cm)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="cifar5_run", help="Root directory containing artifacts folder")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts subfolder name")
    args=ap.parse_args()
    try:
        # Handle single artifacts directory (like cifar5_run/artifacts)
        artifacts_dir = os.path.join(args.root, args.artifacts)
        eval_path = os.path.join(artifacts_dir, "eval.json")
        
        if not os.path.exists(eval_path):
            log.error(f"missing {eval_path}")
            sys.exit(1)
            
        data = json.load(open(eval_path))
        
        # Generate confusion matrix plot
        out_png = os.path.join(artifacts_dir, "confusion_matrix.png")
        plot_cm(data["confusion_matrix"], data["classes"], 
                title=f"Confusion Matrix - {args.root}", out_png=out_png)
        
        # Save classification report
        rpt_path = os.path.join(artifacts_dir, "classification_report.txt")
        with open(rpt_path, "w") as f:
            if isinstance(data["report"], str):
                f.write(data["report"])
            else:
                f.write(json.dumps(data["report"], indent=2))
        
        log.info(f"Wrote {out_png} and {rpt_path}")
        log.info(f"Validation accuracy: {data.get('val_acc', 'N/A')}")
        
    except Exception as e:
        log.exception(e); sys.exit(1)

if __name__=="__main__": main()
