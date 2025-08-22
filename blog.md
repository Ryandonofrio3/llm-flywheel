# The LLM Data Flywheel: Why Your Next ML Model Should Start With Language Models

> Thesis: Use an LLM as a temporary teacher to bootstrap labels, structure messy data, and surface edge cases. Distill into a small, cheap student. Deploy a cascade. Let production feedback restart the loop.

---

## 0) TL;DR

* LLMs create a **data flywheel**: label → distill → deploy → collect errors → relabel → improve.
* Result: lower unit cost, higher coverage, faster iteration.
* Template repo: `rvl_med5/` gate + cost worksheet (link).

---

## 1) The Flywheel At A Glance

```
Spec → LLM labels → QC → Train small model → Calibrate & cascade → Ship → Collect errors → Targeted relabel → ↑
```

* **Spec**: task, schema, metric.
* **LLM labels**: strict JSON, confidence, rationale.
* **QC**: agreement, heuristics, spot-check.
* **Student**: XGBoost/MobileNet/TinyBERT.
* **Cascade**: student first, LLM on uncertain.
* **Feedback**: capture disagreements, drift, novel cases.

---

## 2) When This Beats “LLM Everywhere”

* Latency budgets <50 ms or high QPS.
* Stable schemas.
* Repetitive labeling tasks.
* Predictable costs required.

---

## 3) Three Patterns

1. **Labeler**: LLM tags or scores data; student learns decision boundary.
2. **Normalizer**: LLM maps spans to ontology; student learns embeddings-to-code.
3. **Triage**: LLM creates relevance/severity ranks; student becomes a fast gate.

---

## 4) Running Example (Public, NDA-safe)

**Thumbnail Page-Type Gate (medical-like proxies using RVL-CDIP)**

* Task: classify page thumbnails into {forms, letters, billing, narrative, other}.
* Teacher: Gemini Flash labels a subset with confidence.
* Student: MobileNetV3-Small.
* Outputs: accuracy, confusion matrix, throughput, cost breakeven.
* Repo pieces: `rvlcdip_distill.py`, `plot_confusion.py`, `cost_breakeven.py`, `bench_infer.py`.

---

## 5) Quality Control Recipe

* **Prompt discipline**: fixed schema, examples, definitions.
* **Dual pass agreement**: keep if labels match and confidence ≥ τ.
* **Heuristic vetoes**: regex or rules can invalidate.
* **Gold pins**: 100–300 hand-checked items for sanity.
* **Leakage control**: split by source or entity, not random.

---

## 6) Student Training Template

* Models:

  * Text → **XGBoost** over tf-idf + features.
  * Image → **MobileNetV3-Small** 224², light aug.
* Loss: cross-entropy with label smoothing.
* Calibration: temperature scaling on a held-out set.
* Export: ONNX or TorchScript.

---

## 7) The Cascade

* **Route 1**: student confident → act.
* **Route 2**: student uncertain → escalate to LLM.
* **Route 3**: LLM uncertain → human.
* Log every escalation for the next flywheel pass.

---

## 8) Metrics That Matter

* Label **agreement** (LLM↔LLM, LLM↔mapped).
* Student **macro-F1** or AUROC.
* **Coverage at fixed precision** for gating.
* **Throughput** img/s or docs/s on CPU.
* **Cost curve**: \$/1M items baseline vs flywheel.
* **Breakeven time** for one-time labeling.

---

## 9) Cost Model (plug numbers)

* Gate cost per item: {{gate\_cost}}
* OCR/LLM heavy step: {{heavy\_cost}}
* Label subset fraction: {{label\_frac}} at {{label\_cost}} each
* Monthly volume: {{volume}}
* Output: savings and breakeven (see `cost_breakeven.py`).

---

## 10) Prompts (copy-paste)

**Classifier (5-class)**

```text
You are a precise page-type judge. Return strict JSON only.
Classes: ["forms","letters","billing","narrative","other"].
Definitions: ...
Return: {"label":"<one>","confidence":0-1,"why":"<=15 words"}.
```

**Relevance grader (0–4)**

```text
Score how well the passage answers the query on {0..4}. Return {"score":int,"confidence":0-1,"why":"<=15 words"}.
```

---

## 11) Minimal Pipeline Pseudocode

```python
# 1) sample -> 2) label with LLM (dual-pass) -> 3) filter -> 4) train student -> 5) eval -> 6) cascade
labels = llm_label(sample, schema, agree=True, conf>=0.7)
student = train_small_model(features, labels)
calibrate(student, dev)
deploy(student, escalate_to=LLM, conf_thresh=τ)
collect(disagreements).send_to(llm_label)
```

---

## 12) Case Studies You Can Swap In (one paragraph each)

* **RAG re-ranker**: LLM 0–4 relevance → Mini cross-encoder.
* **Support intent**: LLM taxonomy → DistilBERT or fastText.
* **PII pre-filter**: LLM detects PHI types → XGBoost one-vs-rest.
* **Radiology normal/abnormal**: Report-derived labels → MobileNet.

---

## 13) Failure Modes and Mitigations

* **Drift**: monitor confidence histograms; scheduled relabeling.
* **Schema creep**: version prompts and labels; migration scripts.
* **Bias**: per-slice metrics; active error discovery.
* **Over-trust**: always keep a human-review lane for low-confidence.

---

## 14) Governance and Provenance

* Store: prompt version, model version, label source, confidence, time, data hash.
* Publish a short **datacard** with known limitations.

---

## 15) Results Snapshot (fill after runs)

* Student accuracy: {{acc}}.
* Gate throughput: {{imgs\_per\_s}} img/s on {{CPU}}.
* Savings vs baseline at {{volume}} pages: \${{savings}}/month.
* Breakeven: {{months}} months.

---

## 16) Reproduce

```bash
# build subset, label subset, train three arms
python rvlcdip_distill.py --remap_mode medical5 --drop_unmapped --n_per_class 600 --label_limit 3000 --epochs 5
python plot_confusion.py --root rvl_med5
python bench_infer.py --jsonl rvl_med5/val_final.jsonl --models rvl_med5/artifacts_C_combined --workers 8 --threads_per_worker 2
python cost_breakeven.py --pages 1000000 --gate_acc {{C_acc}}
```

---

## 17) Takeaways

* Start with an LLM to shape the data and the spec.
* Distill quickly into a tiny model.
* Use a cascade to cap cost while keeping quality.
* Treat disagreements as labeled data for the next turn of the flywheel.

---

## 18) Appendix

* Prompt JSON schemas.
* Label acceptance rules.
* Hyperparameters.
* Links to plots and reports.
