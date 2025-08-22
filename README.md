# LLM Data Flywheel

> **Thesis**: Use an LLM as a temporary teacher to bootstrap labels, distill into a small model, and create a self-improving data flywheel.

**The cycle**: LLM labels → Train student → Deploy cascade → Collect errors → Targeted relabel → Improve

## 🎯 Demo: CIFAR5 Classification

This repo demonstrates the flywheel using a 5-class image classification task:
- **Teacher**: Gemini 2.5 Flash vision model via OpenRouter
- **Student**: MobileNet v3 Small (6MB)
- **Result**: 87.4% accuracy, 900+ images/sec on Apple Silicon

## 🚀 Quick Start

```bash
# Setup
git clone <this-repo>
cd llm-as-labelers
uv sync

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Run the full pipeline
uv run cifar_distill.py --step all

# Analyze results
uv run plot_confusion.py
uv run simple_speed_test.py --compare_devices
```

## 📋 Pipeline Steps

### 1. Data Prep
```bash
uv run cifar_distill.py --step prep
```
- Downloads CIFAR-100
- Filters to 5 classes: apple, mushroom, orange, pear, sweet_pepper  
- Upscales to 224x224, applies sharpening
- Saves train (2500) and test (500) images

### 2. LLM Labeling
```bash
uv run cifar_distill.py --step label --max_label 1500
```
- Dual-pass labeling with consistency checks
- Base64 encodes images for vision API
- Progress tracking with resume capability
- Outputs `review_labels.csv` for human QC

### 3. Student Training
```bash
uv run cifar_distill.py --step train
```
- Trains MobileNet on LLM labels
- Data augmentation during training
- Evaluates on original CIFAR test set
- Saves model + confusion matrix

### 4. Analysis
```bash
uv run plot_confusion.py      # Visualize performance
uv run simple_speed_test.py   # Benchmark inference speed
```

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.4% (vs 20% random) |
| **Speed** | 900 img/s on Apple Silicon |
| **Model Size** | 6MB MobileNet |
| **Best Class** | Mushroom (95% F1) |
| **Hardest** | Pear vs Apple confusion |

## 🔄 The Flywheel in Action

1. **Bootstrap**: LLM creates initial training set
2. **Distill**: Train fast, small student model  
3. **Deploy**: Student handles most traffic (cheap)
4. **Cascade**: LLM processes uncertain cases
5. **Feedback**: Collect disagreements and edge cases
6. **Iterate**: Retrain with focused examples

## 🧠 Why This Beats "LLM Everywhere"

- **10x cheaper**: Student inference vs repeated LLM calls
- **100x faster**: Local model vs API latency  
- **Better reliability**: Deterministic student behavior
- **Data leverage**: Each error improves the system
- **Compound gains**: Flywheel accelerates over time

## 📁 Core Files

```
cifar_distill.py      # Main pipeline script
simple_speed_test.py  # Performance benchmarking  
plot_confusion.py     # Results visualization
blog.md              # Detailed explanation
cost_breakeven.py    # Economics analysis
pyproject.toml       # Dependencies
```

## 🔧 Key Implementation Details

- **Dual-pass labeling**: Consistency check prevents hallucinations
- **Progress tracking**: Resume interrupted labeling runs
- **Device optimization**: Auto-detects MPS/CUDA acceleration  
- **Careful API format**: Fixed OpenRouter vision payload bug
- **Production ready**: Error handling, logging, configurability

## 💡 Extending to Your Use Case

1. **Replace data source**: Swap CIFAR for your domain
2. **Adapt prompts**: Modify classification categories/instructions
3. **Choose student**: XGBoost for tabular, BERT for text, etc.
4. **Tune cascade**: Adjust confidence thresholds
5. **Add feedback loop**: Connect production errors to retraining

## 📚 Learn More

- Read the full explanation in [`blog.md`](blog.md)
- See cost analysis in [`cost_breakeven.py`](cost_breakeven.py)
- Explore variations in the experiments directory

---

*Demonstrates LLM → student knowledge distillation for efficient, self-improving ML systems.* 