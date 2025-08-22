# Sample Results

This directory contains example outputs from running the LLM flywheel pipeline on CIFAR5 classification.

## Files

- `eval.json` - Final model performance metrics and confusion matrix
- `classification_report.txt` - Detailed per-class precision/recall/F1 scores

## Key Results

- **87.4% validation accuracy** on 5-class classification
- **4.37x better than random chance** (20%)
- **Best performing class**: Mushroom (95% F1-score)
- **Most challenging**: Pear vs Apple distinction

## How to Generate

Run the full pipeline to reproduce these results:

```bash
export OPENROUTER_API_KEY="your_key_here"
uv run cifar_distill.py --step all
uv run plot_confusion.py
```

The complete outputs will be saved to `cifar5_run/artifacts/` (excluded from git due to size). 