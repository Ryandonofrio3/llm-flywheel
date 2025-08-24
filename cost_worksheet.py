#!/usr/bin/env python3
import argparse, logging, math, json
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("cost")

def main():
    ap=argparse.ArgumentParser(description="Breakeven of LLM-at-runtime vs distilled student")
    ap.add_argument("--monthly_items", type=int, default=1_000_000)
    ap.add_argument("--llm_cost_per_call", type=float, default=0.005, help="$ per item if sent to LLM")
    ap.add_argument("--student_pct_handled", type=float, default=0.9, help="fraction handled by student")
    ap.add_argument("--escalation_cost", type=float, default=0.005, help="$ per escalation to LLM")
    ap.add_argument("--host_cost_per_hour", type=float, default=0.15, help="$ per hr for CPU VM")
    ap.add_argument("--student_ips", type=float, default=1500.0, help="images/sec on chosen host")
    ap.add_argument("--retrain_hours", type=float, default=2.0, help="trainer runtime hours per month")
    ap.add_argument("--retrain_host_cost_per_hour", type=float, default=0.50)
    args=ap.parse_args()

    # LLM runtime plan
    cost_llm = args.monthly_items * args.llm_cost_per_call

    # Student plan
    handled = args.monthly_items * args.student_pct_handled
    escalated = args.monthly_items - handled
    # hosting hours needed
    hours = handled / (args.student_ips * 3600.0)
    host_cost = hours * args.host_cost_per_hour
    esc_cost = escalated * args.escalation_cost
    retrain_cost = args.retrain_hours * args.retrain_host_cost_per_hour
    cost_student = host_cost + esc_cost + retrain_cost

    # Breakeven monthly_items for given params (solve tokens*x = host(x)+esc(x))
    # tokens*x = (x*student_pct/ips/3600)*host + (x*(1-student_pct))*esc + retrain
    a = args.llm_cost_per_call
    b = (args.student_pct_handled/(args.student_ips*3600.0))*args.host_cost_per_hour + (1-args.student_pct_handled)*args.escalation_cost
    x_be = float("inf") if a<=b else (retrain_cost)/(a-b)

    out = {
        "inputs": vars(args),
        "llm_runtime_cost": round(cost_llm, 2),
        "student_monthly_cost": round(cost_student, 2),
        "breakeven_items_per_month": (None if math.isinf(x_be) else int(x_be))
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__": main()
