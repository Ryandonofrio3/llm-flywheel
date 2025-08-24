PY=uv run
OUT=cifar5_run
ART=$(OUT)/artifacts
TEST=$(OUT)/images/test

.PHONY: all prep label train plot bench export one clean

all: prep label train plot bench

prep:
	$(PY) cifar5_llm_distill.py --step prep

label:
	$(PY) cifar5_llm_distill.py --step label --max_label 1500 --conf_thresh 0.0

train:
	$(PY) cifar5_llm_distill.py --step train --epochs 15

plot:
	$(PY) plot_confusion.py --root $(OUT) --artifacts artifacts

bench:
	$(PY) bench_infer.py --data_dir $(TEST) --ckpt $(ART)/mobilenet.pt --batch_size 256

export:
	$(PY) export_model.py --artifacts $(ART)

one:
	$(PY) infer_one.py --artifacts $(ART) --image $(TEST)/apple/apple_00000.jpg

cost:
	$(PY) cost_worksheet.py --scenario

latency:
	$(PY) latency_comparison.py --samples 20

confidence:
	$(PY) confidence_analysis.py --sample-size 200

analyze: cost latency confidence

clean:
	rm -rf $(OUT)/_torchvision __pycache__
