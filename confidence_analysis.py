#!/usr/bin/env python3
"""
Confidence distribution analysis for escalation thresholds
"""
import os, json, argparse, logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("confidence")

class TestDataset:
    def __init__(self, test_dir):
        self.items = []
        self.class_to_idx = {}
        
        classes = ["apple", "mushroom", "orange", "pear", "sweet_pepper"]
        for i, cls in enumerate(classes):
            self.class_to_idx[cls] = i
            cls_dir = os.path.join(test_dir, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith('.jpg'):
                        self.items.append((os.path.join(cls_dir, f), cls))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, true_class = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image)
        y = self.class_to_idx[true_class]
        return x, y, img_path, true_class

def load_model(artifacts_dir):
    """Load trained student model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    eval_data = json.load(open(os.path.join(artifacts_dir, "eval.json")))
    classes = eval_data["classes"]
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    
    state_dict = torch.load(os.path.join(artifacts_dir, "mobilenet.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    
    return model, classes, device

def analyze_confidence_distribution(test_dir, artifacts_dir, output_dir=".", sample_size=None):
    """Analyze confidence distributions and accuracy by confidence bins."""
    
    # Load model and data
    model, classes, device = load_model(artifacts_dir)
    dataset = TestDataset(test_dir)
    
    if sample_size and sample_size < len(dataset):
        # Random sample for faster analysis
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        dataset.items = [dataset.items[i] for i in indices]
    
    log.info(f"Analyzing {len(dataset)} test images...")
    
    # Collect predictions
    predictions = []
    
    with torch.no_grad():
        for i, (x, y_true, img_path, true_class) in enumerate(dataset):
            if i % 100 == 0:
                log.info(f"Processed {i}/{len(dataset)} images...")
                
            x_batch = x.unsqueeze(0).to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            pred_idx = np.argmax(probs)
            max_confidence = probs[pred_idx]
            predicted_class = classes[pred_idx]
            
            is_correct = (predicted_class == true_class)
            
            predictions.append({
                "image_path": img_path,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": max_confidence,
                "is_correct": is_correct,
                "probabilities": dict(zip(classes, probs))
            })
    
    # Convert to arrays for analysis
    confidences = np.array([p["confidence"] for p in predictions])
    correct = np.array([p["is_correct"] for p in predictions])
    
    # Analyze by confidence bins
    bin_edges = np.arange(0, 1.1, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_stats = []
    for i in range(len(bin_edges) - 1):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # Last bin includes 1.0
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        bin_count = mask.sum()
        bin_accuracy = correct[mask].mean() if bin_count > 0 else 0
        
        bin_stats.append({
            "bin_start": bin_edges[i],
            "bin_end": bin_edges[i + 1],
            "bin_center": bin_centers[i],
            "count": int(bin_count),
            "accuracy": bin_accuracy,
            "percentage": bin_count / len(predictions) * 100
        })
    
    # Analyze by class
    class_stats = defaultdict(list)
    for pred in predictions:
        class_stats[pred["true_class"]].append({
            "confidence": pred["confidence"],
            "is_correct": pred["is_correct"]
        })
    
    # Generate visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confidence distribution histogram
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy vs Confidence
    plt.subplot(1, 2, 2)
    bin_centers_plot = [bs["bin_center"] for bs in bin_stats if bs["count"] > 0]
    bin_accuracies = [bs["accuracy"] for bs in bin_stats if bs["count"] > 0]
    bin_counts = [bs["count"] for bs in bin_stats if bs["count"] > 0]
    
    plt.scatter(bin_centers_plot, bin_accuracies, s=[c*2 for c in bin_counts], alpha=0.7)
    plt.plot(bin_centers_plot, bin_accuracies, '--', alpha=0.5)
    plt.xlabel('Confidence Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence by class
    plt.figure(figsize=(12, 8))
    
    # Separate correct and incorrect predictions
    for i, cls in enumerate(classes):
        cls_data = class_stats[cls]
        if not cls_data:
            continue
            
        correct_confs = [d["confidence"] for d in cls_data if d["is_correct"]]
        incorrect_confs = [d["confidence"] for d in cls_data if not d["is_correct"]]
        
        plt.subplot(2, 3, i + 1)
        if correct_confs:
            plt.hist(correct_confs, bins=15, alpha=0.7, label='Correct', color='green')
        if incorrect_confs:
            plt.hist(incorrect_confs, bins=15, alpha=0.7, label='Incorrect', color='red')
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title(f'{cls.title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_by_class.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate escalation scenarios
    escalation_scenarios = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        low_confidence_mask = confidences < threshold
        escalation_rate = low_confidence_mask.mean()
        
        # Accuracy of high-confidence predictions
        high_conf_mask = confidences >= threshold
        high_conf_accuracy = correct[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0
        
        escalation_scenarios.append({
            "threshold": threshold,
            "escalation_rate": escalation_rate,
            "high_confidence_accuracy": high_conf_accuracy,
            "high_confidence_count": int(high_conf_mask.sum()),
            "low_confidence_count": int(low_confidence_mask.sum())
        })
    
    # Print analysis
    print(f"\n📊 CONFIDENCE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Overall accuracy: {correct.mean():.3f}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Median confidence: {np.median(confidences):.3f}")
    print()
    
    print(f"📈 CONFIDENCE BINS")
    print(f"{'Bin':>8} {'Count':>8} {'Pct':>8} {'Accuracy':>10}")
    print(f"{'-'*40}")
    for bs in bin_stats:
        if bs["count"] > 0:
            print(f"{bs['bin_start']:.1f}-{bs['bin_end']:.1f} "
                  f"{bs['count']:>6} "
                  f"{bs['percentage']:>6.1f}% "
                  f"{bs['accuracy']:>9.3f}")
    
    print(f"\n🚨 ESCALATION SCENARIOS")
    print(f"{'Threshold':>10} {'Escalate%':>10} {'Accuracy':>10} {'High-Conf#':>12}")
    print(f"{'-'*45}")
    for scenario in escalation_scenarios:
        print(f"{scenario['threshold']:>9.2f} "
              f"{scenario['escalation_rate']*100:>8.1f}% "
              f"{scenario['high_confidence_accuracy']:>9.3f} "
              f"{scenario['high_confidence_count']:>10}")
    
    # Export results
    results = {
        "overall_stats": {
            "total_predictions": len(predictions),
            "overall_accuracy": float(correct.mean()),
            "mean_confidence": float(confidences.mean()),
            "median_confidence": float(np.median(confidences))
        },
        "confidence_bins": bin_stats,
        "escalation_scenarios": escalation_scenarios,
        "class_stats": {cls: {
            "count": len(data),
            "accuracy": float(np.mean([d["is_correct"] for d in data])),
            "mean_confidence": float(np.mean([d["confidence"] for d in data]))
        } for cls, data in class_stats.items()},
        "detailed_predictions": predictions
    }
    
    # Convert numpy types in detailed_predictions
    for p in results["detailed_predictions"]:
        p["confidence"] = float(p["confidence"])
        p["is_correct"] = bool(p["is_correct"])
        for k, v in p["probabilities"].items():
            p["probabilities"][k] = float(v)

    with open(os.path.join(output_dir, "confidence_analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Analysis complete. Results saved to {output_dir}/")
    log.info(f"Generated: confidence_analysis.png, confidence_by_class.png, confidence_analysis_results.json")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze confidence distributions for escalation thresholds")
    parser.add_argument("--test-dir", default="cifar5_run/images/test",
                       help="Directory with test images")
    parser.add_argument("--artifacts", default="cifar5_run/artifacts",
                       help="Artifacts directory with trained model")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for plots and results")
    parser.add_argument("--sample-size", type=int,
                       help="Random sample size (default: use all images)")
    
    args = parser.parse_args()
    
    analyze_confidence_distribution(args.test_dir, args.artifacts, 
                                  args.output_dir, args.sample_size)

if __name__ == "__main__":
    main() 