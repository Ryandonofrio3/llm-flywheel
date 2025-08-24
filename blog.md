# The LLM Data Flywheel: Why Your Next ML Model Should Start With Language Models

TLDR: LLM's serve as incredibly effective data labelers to boostrap a simpler ML model. Transfer learning from the LLM to a smaller model is a great way to get a lot of bang for your buck.

---

## 0) TL;DR

* LLMs create a **data flywheel**: label → distill → deploy → collect errors → relabel → improve.
* Result: lower unit cost, higher coverage, faster iteration.
* **Demo**: CIFAR5 classification achieving 87.4% accuracy with LLM → MobileNet distillation.
* Template repo with full pipeline + cost analysis.

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

## 4) Running Example: CIFAR5 Image Classification

**Food Classification with Vision LLM → MobileNet Pipeline**

* **Task**: Classify 224x224 images into {apple, mushroom, orange, pear, sweet_pepper}
* **Teacher**: Gemini 2.5 Flash vision model via OpenRouter API
* **Student**: MobileNetV3-Small (6MB model)
* **Data**: CIFAR-100 subset, upscaled with sharpening
* **Results**: 87.4% accuracy, 900+ img/s on Apple Silicon
* **Files**: `cifar_distill.py`, `plot_confusion.py`, `simple_speed_test.py`, `cost_breakeven.py`

---

## 5) Quality Control Recipe

**Our CIFAR5 Implementation:**
* **Prompt discipline**: Fixed JSON schema, specific class definitions
* **Dual pass agreement**: Two parallel API calls, keep only if labels match
* **Confidence filtering**: Configurable threshold (default: accept all agreements)
* **Progress tracking**: Resume interrupted labeling runs, saves every 10 items
* **Error handling**: Exponential backoff for rate limits, retry logic

**General best practices:**
* **Gold standard**: 100–300 hand-checked items for validation
* **Leakage control**: Split by source/entity, not random sampling
* **Schema versioning**: Track prompt changes and label provenance

---

## 6) Student Training Template

**Image Classification (this repo):**
* **Model**: MobileNetV3-Small (6MB, fast inference)
* **Input**: 224x224 RGB images with ImageNet normalization
* **Augmentation**: RandomResizedCrop, rotation, color jitter
* **Loss**: CrossEntropy with 0.05 label smoothing
* **Optimizer**: AdamW with cosine annealing
* **Device**: Auto-detects MPS (Apple Silicon) vs CPU
* **Validation**: Early stopping on held-out CIFAR test set

**Other domains:**
* **Text**: XGBoost over TF-IDF + engineered features
* **Tabular**: XGBoost or LightGBM with categorical encoding

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

**Vision Classifier (CIFAR5)**

```text
Classify this 224x224 photo into one of:
["apple","mushroom","orange","pear","sweet_pepper"].
If fruit vs vegetable is ambiguous, prefer the fruit.
Return JSON only: {"label":"<one>","confidence":0-1,"why":"<=12 words"}.
```

**System prompt:**
```text
You are a precise image labeler. Respond with strict JSON only.
```

**Key details:**
- Base64 image encoding via OpenRouter vision API
- Dual-pass labeling for consistency (agreement required)
- Exponential backoff for rate limiting

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

* **Document classification**: LLM labels page types → MobileNet thumbnail classifier (this repo).
* **RAG re-ranker**: LLM 0–4 relevance scores → Mini cross-encoder for fast ranking.
* **Support intent**: LLM categorizes tickets → DistilBERT or fastText for real-time routing.
* **PII pre-filter**: LLM detects sensitive content → XGBoost binary classifier for preprocessing.
* **Medical imaging**: Radiologist labels → MobileNet for normal/abnormal screening.

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

## 15) Results Snapshot

**CIFAR5 Classification Results:**

* **Student accuracy**: 87.4% (vs 20% random chance)
* **Inference speed**: 895 img/s on Apple Silicon MPS, 85 img/s on CPU  
* **Model size**: 6MB MobileNetV3-Small
* **Best class**: Mushroom (95% F1-score, 97% recall)
* **Hardest distinction**: Pear vs Apple (common confusion)
* **Training data**: 1,499 LLM-labeled images (dual-pass filtered)
* **LLM cost**: ~$3 for complete labeling run

---

## 16) Reproduce

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Run full pipeline: data prep → LLM labeling → training
uv run cifar_distill.py --step all --max_label 1500 --epochs 30

# Analyze results
uv run plot_confusion.py                    # Confusion matrix visualization
uv run simple_speed_test.py --compare_devices  # Speed benchmark CPU vs MPS
uv run cost_breakeven.py --volume 1000000   # Economics analysis

# Individual steps (optional)
uv run cifar_distill.py --step prep         # Download CIFAR-100, prep images
uv run cifar_distill.py --step label        # LLM labeling only  
uv run cifar_distill.py --step train        # Train student only
```

---

## 17) Key Implementation Lessons

**Critical bug we fixed:** Initially used wrong OpenRouter API format (`"input_image"` instead of `"image_url"`), causing the LLM to never actually see images → labeled everything as "apple" → 20% validation accuracy. Always verify your API calls work!

**Performance insights:**
* Apple Silicon MPS acceleration: 10.5x faster than CPU (895 vs 85 img/s)
* Dual-pass labeling: Worth the 2x API cost for consistency
* Progress tracking: Essential for long labeling runs (1500 images takes ~10 minutes)
* Batch processing: Parallel API calls significantly speed up labeling

**Quality control wins:**
* Dual-pass agreement filtering eliminated hallucinations
* Base64 image encoding worked reliably for vision API
* Label normalization caught common variants (apples→apple, sweet_peppers→sweet_pepper)

## 18) Takeaways

* Start with an LLM to shape the data and the spec.
* Distill quickly into a tiny model.
* Use a cascade to cap cost while keeping quality.
* Treat disagreements as labeled data for the next turn of the flywheel.
* **Verify your API integration thoroughly** - small bugs can break the entire pipeline.

---

## 19) Appendix: Technical Details

**Hyperparameters:**
```python
# LLM Labeling
MODEL = "google/gemini-2.5-flash"
CONFIDENCE_THRESHOLD = 0.0  # Accept all dual-pass agreements
MAX_WORKERS = 10  # Parallel labeling threads
DUAL_PASS = True  # Two API calls per image

# Training  
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-2
AUGMENTATION = True  # RandomCrop, rotation, color jitter
```

**Cost breakdown (1500 images):**
* LLM labeling: ~$3 total (2x calls per image)
* Training time: ~30 minutes on Apple Silicon
* Final model: 6MB, runs at 900+ img/s

**Files generated:**
* `labels_llm.csv` - Raw LLM outputs
* `review_labels.csv` - Human-editable labels  
* `eval.json` - Final metrics + confusion matrix
* `mobilenet.pt` - Trained model weights
