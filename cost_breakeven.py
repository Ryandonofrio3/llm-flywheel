#!/usr/bin/env python3
import os, sys, json, math, logging, argparse, csv
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cost")

def compute_cost(pages:int, class_mix:dict, gate_acc:float, ocr_cost:float,
                 llm_cost_per_img:float, gate_cost_per_img:float, llm_label_cost_per_img:float,
                 label_fraction:float, scenario:str):
    # After training, inference uses gate_cost_per_img per page.
    # Pages predicted as "skip-eligible" avoid OCR cost. Use mix and gate accuracy to estimate skip rate.
    # Assume skip OCR for classes in {"forms","letters","billing"} if you choose; configurable.
    skip_classes = {"forms","letters","billing"}
    mix_skip = sum(class_mix.get(c,0) for c in skip_classes)
    expected_skip = gate_acc * mix_skip  # conservative: treat only correct skips
    cost_infer = pages * gate_cost_per_img + pages * (1 - expected_skip) * ocr_cost
    # LLM baseline (no gate): OCR everything + optional per-page LLM? For comparison, show OCR-all.
    cost_baseline = pages * ocr_cost
    # Labeling one-time cost
    labeled = int(pages * label_fraction)
    label_cost = labeled * llm_label_cost_per_img
    return {
        "scenario": scenario,
        "pages": pages,
        "gate_accuracy": gate_acc,
        "mix_skip": mix_skip,
        "expected_skip_rate": expected_skip,
        "cost_inference_total": round(cost_infer, 4),
        "cost_baseline_ocr_all": round(cost_baseline, 4),
        "labeling_one_time": round(label_cost, 4),
        "monthly_savings_vs_baseline": round(cost_baseline - cost_infer, 4),
        "breakeven_months": round((label_cost / max(1e-9, (cost_baseline - cost_infer))), 4)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1_000_000)
    ap.add_argument("--class_mix_json", default='{"forms":0.35,"letters":0.15,"billing":0.20,"narrative":0.25,"other":0.05}')
    ap.add_argument("--gate_acc", type=float, default=0.95)
    ap.add_argument("--ocr_cost", type=float, default=0.0025)  # $/page OCR
    ap.add_argument("--gate_cost", type=float, default=0.00002)  # $/image CPU amortized
    ap.add_argument("--llm_label_cost", type=float, default=0.002)  # $/image to label subset
    ap.add_argument("--label_fraction", type=float, default=0.003)  # 0.3% labeled once
    ap.add_argument("--out_csv", default="cost_summary.csv")
    args=ap.parse_args()
    try:
        mix=json.loads(args.class_mix_json)
        rows=[]
        for scenario,acc in [("A_mapped_only",args.gate_acc-0.02),("B_llm_only",args.gate_acc),("C_combined",args.gate_acc+0.005)]:
            rows.append(compute_cost(args.pages, mix, max(0.0,min(1.0,acc)), args.ocr_cost,
                                     llm_cost_per_img=0.0, gate_cost_per_img=args.gate_cost,
                                     llm_label_cost_per_img=args.llm_label_cost,
                                     label_fraction=args.label_fraction, scenario=scenario))
        with open(args.out_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
        log.info(f"Wrote {args.out_csv}")
        print(json.dumps(rows, indent=2))
    except Exception as e:
        log.exception(e); sys.exit(1)

if __name__=="__main__": 
    main()
