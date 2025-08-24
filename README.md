# LLM as Labelers

Use an LLM to create training labels, then distill that knowledge into a smaller, faster model.

## What this does

This repo shows how to use Gemini 2.5 Flash to label images from CIFAR-100, then train a small MobileNet model on those labels. The idea is that you get most of the LLM's accuracy but with much faster inference.

## Setup

```bash
git clone <this-repo>
cd llm-as-labelers
uv sync
export OPENROUTER_API_KEY="your_key_here"
```

## Usage

Run the whole pipeline:
```bash
uv run cifar_distill.py --step all
```

Or run individual steps:
```bash
uv run cifar_distill.py --step prep    # Download and prepare CIFAR data
uv run cifar_distill.py --step label   # Get LLM labels
uv run cifar_distill.py --step train   # Train student model
```

## What it does

1. Takes CIFAR-100 and filters it down to 5 classes: apple, mushroom, orange, pear, sweet_pepper
2. Sends images to Gemini 2.5 Flash for labeling (with dual-pass consistency checking)
3. Trains a MobileNet v3 Small on those labels
4. Evaluates on the original CIFAR test set

## Results

The student model gets about 87% accuracy and runs at 900+ images/second on Apple Silicon. The model file is only 6MB.

## Files

- `cifar_distill.py` - Main script that does everything
- `plot_confusion.py` - Visualize the confusion matrix
- `throughput.py` - Test inference speed
- `blog.md` - Longer explanation of the approach

## Why this matters

Instead of calling an expensive LLM API for every prediction, you can:
1. Use the LLM once to create a training set
2. Train a small model that captures most of the LLM's knowledge
3. Deploy the small model for fast, cheap inference
4. Fall back to the LLM only for uncertain cases

This is especially useful when you need to classify thousands of items quickly or want to run inference locally. 