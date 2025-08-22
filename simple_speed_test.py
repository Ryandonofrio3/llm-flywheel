#!/usr/bin/env python3
import os
import time
import torch
import glob
from PIL import Image
from torchvision import transforms, models
import json

def load_model(artifacts_dir, device):
    """Load the trained CIFAR5 model."""
    # Load classes
    eval_path = os.path.join(artifacts_dir, "eval.json")
    with open(eval_path, 'r') as f:
        data = json.load(f)
    classes = data["classes"]
    
    # Load model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    
    ckpt_path = os.path.join(artifacts_dir, "mobilenet.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval().to(device)
    return model, classes

def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def simple_speed_test(artifacts_dir="cifar5_run/artifacts", 
                     images_dir="cifar5_run/images/test",
                     device="cpu", 
                     batch_size=32, 
                     max_images=100,
                     warmup_batches=5):
    """Simple, direct speed test."""
    
    print(f"🚀 Simple Speed Test")
    print(f"Device: {device}, Batch Size: {batch_size}, Max Images: {max_images}")
    
    # Load model
    print("Loading model...")
    model, classes = load_model(artifacts_dir, device)
    transform = get_transform()
    
    # Get image paths
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i in range(warmup_batches):
            dummy_batch = torch.randn(batch_size, 3, 224, 224).to(device)
            _ = model(dummy_batch)
    
    # Real test
    print("Running speed test...")
    total_images = 0
    total_time = 0.0
    batch_times = []
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = transform(img)
                    images.append(tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                continue
                
            batch_tensor = torch.stack(images).to(device)
            
            # Time inference only
            torch.cuda.synchronize() if device.startswith("cuda") else None
            start_time = time.perf_counter()
            
            logits = model(batch_tensor)
            predictions = torch.argmax(logits, dim=1)
            
            torch.cuda.synchronize() if device.startswith("cuda") else None
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_time += batch_time
            total_images += len(images)
            
            if (i // batch_size + 1) % 10 == 0:
                current_fps = total_images / total_time
                print(f"  Processed {total_images} images, {current_fps:.1f} img/s")
    
    # Results
    avg_fps = total_images / total_time if total_time > 0 else 0
    avg_batch_time_ms = (sum(batch_times) / len(batch_times)) * 1000 if batch_times else 0
    avg_per_image_ms = avg_batch_time_ms / batch_size if batch_size > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total inference time: {total_time:.3f}s")
    print(f"Average throughput: {avg_fps:.1f} images/second")
    print(f"Average batch time: {avg_batch_time_ms:.2f}ms")
    print(f"Average per image: {avg_per_image_ms:.2f}ms")
    
    return {
        "device": device,
        "batch_size": batch_size,
        "total_images": total_images,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "avg_batch_time_ms": avg_batch_time_ms,
        "avg_per_image_ms": avg_per_image_ms
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--compare_devices", action="store_true", help="Test both CPU and MPS")
    args = parser.parse_args()
    
    if args.compare_devices:
        print("🔥 Device Comparison\n")
        
        # Test CPU
        cpu_results = simple_speed_test(device="cpu", batch_size=args.batch_size, max_images=args.max_images)
        
        print("\n" + "="*50 + "\n")
        
        # Test MPS if available
        if torch.backends.mps.is_available():
            mps_results = simple_speed_test(device="mps", batch_size=args.batch_size, max_images=args.max_images)
            
            print(f"\n🏆 Winner: ", end="")
            if mps_results["avg_fps"] > cpu_results["avg_fps"]:
                speedup = mps_results["avg_fps"] / cpu_results["avg_fps"]
                print(f"MPS ({speedup:.1f}x faster)")
            else:
                speedup = cpu_results["avg_fps"] / mps_results["avg_fps"]
                print(f"CPU ({speedup:.1f}x faster)")
        else:
            print("MPS not available on this system")
    else:
        simple_speed_test(device=args.device, batch_size=args.batch_size, max_images=args.max_images) 