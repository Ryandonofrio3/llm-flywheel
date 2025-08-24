The LLM Flywheel: Use AI to Teach, Not to Serve
You need to process millions of items. Calling an LLM API for each one isn't realistic. At 0.6 requests per second, you'd be waiting until 2030.
But instead of trying to make Gemini run faster, what if the LLM didn't need to do the actual work?
LLMs Are Teachers, Not Workers
Here's the pattern: use the LLM to create a training dataset, then train something small and fast to handle production traffic.
The LLM flywheel:

LLM labels your data (expensive, slow, smart)
You train a tiny model on those labels (cheap, fast, good enough)
Deploy the small model for real traffic
Route the weird edge cases back to the LLM
Use those examples to keep training your model
Watch it get better over time

Testing the Idea
I built a simple proof of concept to show this works. Took CIFAR-100 images, filtered down to 5 classes (apple, mushroom, orange, pear, sweet pepper), and had Gemini 2.5 Flash label about 1,500 images.
Cost me sixty cents.
Then I trained a MobileNetV3-Small model on those labels. Took about 5 minutes on my MacBook.
The results were pretty wild:
My little student model:
- **8.5ms** average latency
- **117 images per second** (single thread)
- **91% accuracy**
Gemini API:
- **1.6 seconds** average latency (1468ms)
- **0.7 images per second**
That's **172x faster**. Not 72% faster. *One hundred and seventy-two times faster.*
Here is the confusion matrix showing the model's performance. It correctly classifies most images, with pears being the most commonly confused class.
![Confusion Matrix](cifar5_run/artifacts/confusion_matrix.png)
The Magic Is in the Routing
The real insight is using confidence scores to decide when to stay local vs. ask the LLM:

Model confident (>85%)? Use the local result
Model unsure? Ask Gemini and add that example to your training set

In practice, this cuts your API calls by about 90% while keeping accuracy close to the original LLM.
My `confidence_analysis.py` script makes this concrete. It shows how overall accuracy changes as you escalate more of the low-confidence predictions to the LLM. If we set a threshold of 0.80, we only need to escalate 37% of images to achieve 100% accuracy on the remaining high-confidence predictions.
```
🚨 ESCALATION SCENARIOS
 Threshold  Escalate%   Accuracy   High-Conf#
---------------------------------------------
     0.50      10.5%     0.939        179
     0.60      16.5%     0.952        167
     0.70      21.0%     0.975        158
     0.80      36.0%     0.984        128
     0.85      41.5%     0.991        117
     0.90      54.5%     1.000         91
     0.95      78.5%     1.000         43
```
And here's the flywheel part: your model keeps getting better because it's constantly learning from the hardest cases. The ones where it was wrong or uncertain get sent back to the teacher.
The Details Matter: From Idea to Robust Pipeline
Getting this to work reliably required more than just a simple script. Here are a few details from the implementation that made a real difference:

- **Two-Pass Consensus**: I didn't trust a single LLM call. For each image, I asked the LLM for a label *twice* in parallel. If the labels didn't match, I threw the result away. This simple consensus step significantly improves label quality.
- **Graceful Error Handling**: APIs fail. The script is built to handle this, with exponential backoff and retries for things like rate limits or temporary server errors.
- **Resume-ability**: The labeling job for 1,500 images takes time. The script saves its progress continuously, so it can be stopped and restarted without losing work or re-processing (and re-paying for) images.
- **Human in the Loop**: The LLM's labels aren't perfect. The script outputs a `review_labels.csv` file. This gives you a chance to manually scan and correct the labels before kicking off the training process, giving you final control over the dataset.

Why This Actually Matters
This isn't just about saving money on API calls. Though at millions of requests, it definitely saves money.
It's about control. Your model weights don't change unless you want them to. No rate limits. No data leaving your servers. You can run it offline, batch it, quantize it, whatever.

But let's be clear: it's also about the money. A lot of money.
After correcting my initial analysis to use token-based pricing for the LLM and a realistic serverless cost model for the student, the savings are staggering. The student model isn't just cheaper at massive scale; it's orders of magnitude cheaper from the start.

Here’s a scenario analysis. Even with a 10% escalation rate (meaning 10% of requests still go to the expensive LLM), the savings are around 90%.

```
📊 COST SCENARIO: 10% Escalation Rate
   Volume/Mo     LLM-Only      Student      Savings     Save %
-----------------------------------------------------------------
      10,000 $       2.10 $       0.21 $       1.89      89.9%
     100,000 $      21.00 $       2.13 $      18.87      89.9%
   1,000,000 $     210.00 $      21.31 $     188.69      89.9%
  10,000,000 $    2100.00 $     213.10 $    1886.90      89.9%
```

The point is this: you get a **172x speedup** at a **~90% discount**, while also gaining control, privacy, and predictability.

I've seen this pattern work for document classification, content moderation, data extraction—basically anywhere you need LLM-quality results at real-world scale.
The key realization: most production AI problems don't need the full power of a frontier model for every single request. They need it for the training data and the edge cases.
Try It
The full code is on GitHub. You can run the whole process yourself.
```bash
git clone https://github.com/your-repo/llm-as-labelers
uv sync  
export OPENROUTER_API_KEY=your-key
make all
```

The `make all` command runs the entire pipeline:
1.  `make prep`: Downloads and prepares the CIFAR image subset.
2.  `make label`: Uses the Gemini API to label 1,500 training images (this is the slow, expensive part).
3.  `make train`: Fine-tunes the MobileNet model on the LLM-generated labels.
4.  `make plot`: Generates a confusion matrix from the evaluation results.
5.  `make bench`: Benchmarks the final model's latency and throughput.

The repository also includes scripts to reproduce the cost and latency analysis from this post. Check out the `Makefile` for more.
The specific example is toy data, but the pattern works at scale. I've used variations of this in production for problems way bigger than fruit classification.
The breakthrough isn't technical—it's architectural. Stop trying to make LLMs faster. Use them to make something else smarter.

Results here are from a synthetic example, but I've applied this pattern successfully in production at much larger scale.