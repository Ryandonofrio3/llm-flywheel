#!/usr/bin/env python3
"""
Latency comparison: LLM API vs Local Student Model
"""
import os, time, json, argparse, logging, statistics
import requests, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("latency")

load_dotenv()

class StudentModel:
    def __init__(self, artifacts_dir):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.classes = self._load_model(artifacts_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        
        # Warmup
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)
    
    def _load_model(self, artifacts_dir):
        eval_data = json.load(open(os.path.join(artifacts_dir, "eval.json")))
        classes = eval_data["classes"]
        
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        
        state_dict = torch.load(os.path.join(artifacts_dir, "mobilenet.pt"), 
                               map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model, classes
    
    def predict(self, image_path):
        """Predict single image and return latency."""
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        latency = time.time() - start_time
        
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        
        return {
            "prediction": self.classes[pred_idx],
            "confidence": confidence,
            "latency_ms": latency * 1000
        }

def llm_predict(image_path, api_key, model="google/gemini-2.5-flash"):
    """Call LLM API and return latency."""
    prompt = (
        "Classify this 224x224 photo into one of: "
        '["apple","mushroom","orange","pear","sweet_pepper"]. '
        'Return JSON only: {"label":"<one>","confidence":0-1}.'
    )
    
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 100,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a precise image labeler. Respond with strict JSON only."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    start_time = time.time()
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        msg = response.json()["choices"][0]["message"]["content"]
        data = json.loads(msg)
        latency = time.time() - start_time
        
        return {
            "prediction": data.get("label", "unknown"),
            "confidence": data.get("confidence", 0.0),
            "latency_ms": latency * 1000
        }
    except Exception as e:
        latency = time.time() - start_time
        log.warning(f"LLM prediction failed: {e}")
        return {
            "prediction": "error",
            "confidence": 0.0,
            "latency_ms": latency * 1000,
            "error": str(e)
        }

def run_comparison(test_images, api_key, artifacts_dir, num_samples=20):
    """Run latency comparison between student model and LLM."""
    
    log.info("Loading student model...")
    student = StudentModel(artifacts_dir)
    
    # Select random sample of images
    import random
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    log.info(f"Running latency comparison on {len(sample_images)} images...")
    
    # Test student model
    log.info("Testing student model latency...")
    student_results = []
    for img_path in sample_images:
        result = student.predict(img_path)
        student_results.append(result)
        log.debug(f"Student: {os.path.basename(img_path)} -> {result['prediction']} "
                 f"({result['latency_ms']:.1f}ms)")
    
    # Test LLM (sequential to avoid rate limits)
    log.info("Testing LLM API latency...")
    llm_results = []
    for i, img_path in enumerate(sample_images):
        if i > 0:
            time.sleep(1)  # Rate limiting
        result = llm_predict(img_path, api_key)
        llm_results.append(result)
        log.debug(f"LLM: {os.path.basename(img_path)} -> {result['prediction']} "
                 f"({result['latency_ms']:.1f}ms)")
    
    return student_results, llm_results

def analyze_results(student_results, llm_results):
    """Analyze and print latency comparison results."""
    
    # Extract latencies
    student_latencies = [r['latency_ms'] for r in student_results]
    llm_latencies = [r['latency_ms'] for r in llm_results if 'error' not in r]
    
    # Student stats
    student_mean = statistics.mean(student_latencies)
    student_p50 = statistics.median(student_latencies)
    student_p95 = statistics.quantiles(student_latencies, n=20)[18] if len(student_latencies) >= 20 else max(student_latencies)
    
    # LLM stats
    if llm_latencies:
        llm_mean = statistics.mean(llm_latencies)
        llm_p50 = statistics.median(llm_latencies)
        llm_p95 = statistics.quantiles(llm_latencies, n=20)[18] if len(llm_latencies) >= 20 else max(llm_latencies)
    else:
        llm_mean = llm_p50 = llm_p95 = float('inf')
    
    # Print results
    print(f"\n🚀 LATENCY COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(student_results)} student, {len(llm_latencies)} LLM successful")
    print()
    
    print(f"📱 STUDENT MODEL (Local)")
    print(f"Mean latency:   {student_mean:>8.1f} ms")
    print(f"P50 latency:    {student_p50:>8.1f} ms") 
    print(f"P95 latency:    {student_p95:>8.1f} ms")
    print()
    
    print(f"🌐 LLM API (Remote)")
    print(f"Mean latency:   {llm_mean:>8.1f} ms")
    print(f"P50 latency:    {llm_p50:>8.1f} ms")
    print(f"P95 latency:    {llm_p95:>8.1f} ms")
    print()
    
    if llm_latencies:
        print(f"⚡ SPEEDUP")
        print(f"Mean speedup:   {llm_mean/student_mean:>8.1f}x")
        print(f"P50 speedup:    {llm_p50/student_p50:>8.1f}x")
        print(f"P95 speedup:    {llm_p95/student_p95:>8.1f}x")
    
    # Throughput estimates
    student_rps = 1000 / student_mean if student_mean > 0 else 0
    llm_rps = 1000 / llm_mean if llm_mean > 0 and llm_mean != float('inf') else 0
    
    print()
    print(f"📊 THROUGHPUT (single thread)")
    print(f"Student model:  {student_rps:>8.1f} req/sec")
    print(f"LLM API:        {llm_rps:>8.1f} req/sec")
    
    # Return summary for export
    return {
        "student": {
            "mean_ms": student_mean,
            "p50_ms": student_p50,
            "p95_ms": student_p95,
            "rps": student_rps
        },
        "llm": {
            "mean_ms": llm_mean,
            "p50_ms": llm_p50,
            "p95_ms": llm_p95,
            "rps": llm_rps
        },
        "speedup": {
            "mean": llm_mean/student_mean if student_mean > 0 and llm_mean != float('inf') else 0,
            "p50": llm_p50/student_p50 if student_p50 > 0 and llm_p50 != float('inf') else 0,
            "p95": llm_p95/student_p95 if student_p95 > 0 and llm_p95 != float('inf') else 0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Compare latency: LLM API vs Student Model")
    parser.add_argument("--test-dir", default="cifar5_run/images/test",
                       help="Directory with test images")
    parser.add_argument("--artifacts", default="cifar5_run/artifacts", 
                       help="Artifacts directory with trained model")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of images to test (default: 20)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("No API key provided. Use --api-key or set OPENROUTER_API_KEY")
        return 1
    
    # Collect test images
    test_images = []
    for cls in ["apple", "mushroom", "orange", "pear", "sweet_pepper"]:
        cls_dir = os.path.join(args.test_dir, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.jpg'):
                    test_images.append(os.path.join(cls_dir, f))
    
    if not test_images:
        log.error(f"No test images found in {args.test_dir}")
        return 1
    
    log.info(f"Found {len(test_images)} test images")
    
    # Run comparison
    student_results, llm_results = run_comparison(test_images, api_key, args.artifacts, args.samples)
    
    # Analyze and export results
    summary = analyze_results(student_results, llm_results)
    
    # Export detailed results
    with open("latency_comparison_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "student_results": student_results,
            "llm_results": llm_results
        }, f, indent=2)
    
    log.info("Results saved to latency_comparison_results.json")

if __name__ == "__main__":
    main() 