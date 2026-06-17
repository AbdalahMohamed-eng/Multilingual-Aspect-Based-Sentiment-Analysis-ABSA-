# DeepX ABSA Solution — Beginner Walkthrough

A step-by-step explanation of how we built an Aspect-Based Sentiment Analysis system for Arabic + English reviews.

---

## What the problem is

We have reviews of businesses (restaurants, apps, shops). Each review is written in Arabic or English. For each review, we need to figure out two things:

1. **What aspects does the review talk about?** From a fixed list of 9: `food`, `service`, `price`, `cleanliness`, `delivery`, `ambiance`, `app_experience`, `general`, `none`.
2. **What's the sentiment for each aspect?** `positive`, `negative`, or `neutral`.

The tricky part is that one review can talk about multiple aspects with different sentiments. Example: "The food was great but the service was slow" → `food=positive`, `service=negative`.

This is called **Aspect-Based Sentiment Analysis (ABSA)**. The evaluation metric is micro F1 on (aspect, sentiment) pairs — both have to match to count as correct.

---

## Step 1 — Look at the data first (EDA)

Before writing any model, always explore the data. This is the step that distinguishes people who win from people who don't.

What we found:

- **Train: 1,971 reviews. Test: 500 reviews.** Not much training data.
- **Train is 98% Arabic. Test is 64% Arabic, 34% English.** Huge mismatch. Our model has to handle both languages despite barely seeing English in training.
- **Test reviews are much shorter** (median 30 chars vs 75 in train). Lots of one-word reviews like "Good" or "ممتاز" (excellent).
- **Star rating is a strong hint.** 5-star reviews are 91% positive aspects; 1-star are 96% negative. The model should definitely use this.
- **Aspect frequencies are unbalanced.** `service` makes up 30% of labels, `none` only 1.7%. Common classes dominate training.

These findings shape every decision that follows.

---

## Step 2 — Pick a model

We need a model that understands both Arabic and English. Options:

- Arabic-only models (MARBERT, CAMeLBERT) → would fail on 34% of test.
- English-only models (BERT, RoBERTa) → would fail on 64% of test.
- **Multilingual models (XLM-RoBERTa)** → understands both. Winner.

We started with **XLM-RoBERTa-base** (278M parameters) for V1, then upgraded to **XLM-RoBERTa-large** (560M parameters) for V2.2. Larger model = more capacity to learn patterns, but slower to train.

These models come pre-trained on web text in 100 languages. We take the pre-trained model and "fine-tune" it on our specific task — much easier than training from scratch.

---

## Step 3 — Design the model architecture

The XLM-RoBERTa backbone gives us a 1024-dimensional vector for each review — a summary of its meaning. We then add two "heads" (small neural networks) on top:

**Head 1: Aspect head.** 9 sigmoid outputs, one per aspect. Each outputs a probability between 0 and 1: "is this aspect mentioned?" For example, if the model outputs `[0.1, 0.9, 0.05, ...]`, that means `food=no`, `service=yes`, `price=no`, etc.

**Head 2: Sentiment head.** For each of the 9 aspects, 3 softmax outputs for positive/negative/neutral. So it's a 9×3 grid of probabilities.

Why two heads? It cleanly separates the two sub-problems. The aspect head learns "is this mentioned?", the sentiment head learns "what's the polarity?". During training, we only punish the sentiment head for aspects that are actually present in the label (this is called "loss masking").

---

## Step 4 — Tell the model the metadata

The review text alone isn't enough. The star rating is a huge hint. So we **prepend the metadata to the review text** before feeding it to the model:

```
[STARS=5] [CAT=مطعم] [PLAT=google_maps] الطعام كان رائعاً
```

Now the model sees the star rating and can learn "when stars=5, aspects tend to be positive." Simple but effective.

---

## Step 5 — Choose the loss function

The loss tells the model how wrong it is. For each review:

- **Aspect loss** = Binary Cross-Entropy (BCE) over the 9 aspects. "Did you predict the right aspects?"
- **Sentiment loss** = Cross-Entropy (CE) over the 3 classes per present aspect. "Did you predict the right sentiment?"

Combined loss = BCE + CE. The model adjusts its weights to minimize this.

**V1 used uniform weighting.** This turned out to be a problem — the common aspects (service, food) dominated the gradient, and the rare ones (cleanliness, none, delivery) were never learned. V1's `cleanliness` F1 was literally 0.

**V2.2 added class weights.** Rare classes get higher loss weight. Specifically, `pos_weight = sqrt((N-positives)/positives)`. This gives `none` 5.8× weight, `cleanliness` 3.1× weight, while `service` stays at 1.0×. Now the rare classes get real gradient signal and actually get learned.

---

## Step 6 — Cross-validation (the trust layer)

We can't just train on all 1,971 rows and check performance on the same 1,971 rows — the model would just memorize them and tell us nothing. We need to test on data the model hasn't seen.

**5-fold cross-validation:** split train into 5 chunks. Train on 4, predict on the 5th. Repeat 5 times, each chunk gets to be the held-out one exactly once. Now every training row has a prediction made by a model that never saw it. These are called **Out-Of-Fold (OOF) predictions**, and they're our honest performance estimate.

We also **stratify** the split by `(number of aspects, star rating, language)` so every fold has a mix of easy/hard, Arabic/English, short/long reviews.

This is where we get the number that matters: **OOF F1 = 0.7916** for V2.2.

---

## Step 7 — Ensemble: train more than one model

Training is random. Different random initialization → slightly different model → slightly different predictions. If you train the same model 10 times and average their predictions, the result is more stable and usually better.

**We trained 10 models:** 5 folds × 2 random seeds per fold. At prediction time, all 10 predict on the test set, and we average their outputs. Cheap way to squeeze out +0.5–1 F1.

---

## Step 8 — Post-processing (decoding)

The model outputs **probabilities**, not predictions. We need rules to turn probabilities into the final JSON.

**Rule 1: Threshold each aspect probability.** If `P(food) > threshold_food`, predict food. Default threshold is 0.5, but different aspects have different optimal thresholds because of class imbalance. We **tune thresholds on OOF predictions** using a simple loop (coordinate ascent) that tries values from 0.1 to 0.8 for each aspect and keeps whatever maximizes F1.

**Rule 2: `none` is exclusive.** `none` means "no specific aspect is discussed." If the model predicts `none` AND `food`, that's contradictory — we drop `none`.

**Rule 3: Empty prediction fallback.** If no aspect crosses its threshold, we'd output an empty list — zero F1 guaranteed. Instead, we take the aspect with the highest probability (argmax) and use star rating to set sentiment (5 stars → positive, 1–2 stars → negative, 3 stars → neutral).

**Rule 4: Conservative neutral.** Neutral is only 4.5% of labels. If the model is unsure, guessing neutral is usually wrong. We only predict neutral when `P(neutral) > 0.80`; otherwise take the better of positive/negative.

---

## Step 9 — Evaluation

We compute **tuple micro F1**: treat each (aspect, sentiment) pair as a unit. Count true positives (we got it exactly right), false positives (we predicted a pair that wasn't there), false negatives (we missed a pair). `F1 = 2 · precision · recall / (precision + recall)`.

For V2.2 we got:

- **OOF F1: 0.7916** (the honest estimate)
- Aspect-only F1: 0.8502 (if we ignore sentiment, how well do we find aspects)
- Sentiment-given-correct-aspect accuracy: 0.9311 (when we get the aspect right, how often is sentiment right)

The decomposition tells us the sentiment head is near-ceiling (93%), so remaining improvements should focus on the aspect head.

---

## Step 10 — Inference on new data

For the competition, we predict on the 500 hidden test reviews and write a JSON file. The format is:

```json
[
  {
    "review_id": 23,
    "aspects": ["food", "service"],
    "aspect_sentiments": {"food": "positive", "service": "negative"}
  },
  ...
]
```

For general use, we saved the 10 trained models to disk (~11 GB total in fp16). `inference.py` loads them, runs any new review through all 10, averages their outputs, applies the tuned thresholds and post-processing, and writes the JSON. Anyone with the checkpoints and the inference script can now predict on new data without retraining.

---

## The iteration story (the most useful lesson)

This is how real ML work goes. Not one shot, multiple tries:

| Version | Change | Result |
|---|---|---|
| **V1** | XLM-R-base, uniform loss, 4 epochs | **OOF F1 = 0.6161.** Good baseline. But `cleanliness`, `none`, `delivery` had F1 = 0.000 — the model literally never predicted them. |
| **V2** | Tried mDeBERTa-v3-base + class weights | Training blew up with NaN loss due to numerical instability in mDeBERTa's fp32 implementation. Dead run. **Lesson: a known-working backbone beats a "theoretically better" unstable one.** |
| **V2.1** | XLM-R-large + all V2 improvements | Trained fine, but we discovered we had been predicting on the wrong file (`DeepX_unlabeled.xlsx` was an unlabeled pool, not the graded set). |
| **V2.2** | Same as V2.1 but pointing at correct test file | **OOF F1 = 0.7916.** +17.5 points over V1. Dead rare classes now have F1 of 0.60, 0.68, 0.41. Class weighting was the single biggest win. |
| **V2.3** | Same training as V2.2, but saves model checkpoints | Same F1, but now inference can be re-run on new data without retraining. |

---

## The key principles worth internalizing

1. **Always do EDA first.** The language mismatch, length mismatch, and star-rating correlation shaped every decision.
2. **Honest evaluation matters more than clever modeling.** OOF F1 is the number to trust. A validation set that matches training distribution will lie to you.
3. **Diagnose before scaling.** When V1 had dead rare classes, a bigger model wouldn't help; class weighting did. Always look at per-class metrics, not just the aggregate.
4. **Simple post-processing can add +5 F1** (threshold tuning, exclusivity rules, fallbacks). Don't skip this step to chase architectural wins.
5. **Iterate.** V1 → V2 → V2.1 → V2.2 → V2.3. Each version has one clear hypothesis, one change, and a measurable result. That's the workflow.
6. **Save the training recipe (code + seeds + config) before saving weights.** Reproducibility is an insurance policy.

---

## Glossary (for quick reference)

- **ABSA** — Aspect-Based Sentiment Analysis. Find aspects and their sentiments in a review.
- **Backbone** — the main pre-trained transformer (XLM-RoBERTa) that turns text into vectors.
- **Head** — a small neural network layer on top of the backbone that produces the final output.
- **Fine-tuning** — taking a pre-trained model and training it a bit more on your specific task.
- **Sigmoid** — squashes any number to a probability between 0 and 1. Used for yes/no decisions.
- **Softmax** — turns a list of numbers into probabilities that sum to 1. Used for pick-one decisions.
- **BCE (Binary Cross-Entropy)** — loss for yes/no predictions.
- **CE (Cross-Entropy)** — loss for pick-one-from-many predictions.
- **pos_weight** — how much extra to penalize the model for missing positive examples. Helps with imbalanced classes.
- **Cross-validation** — splitting data into chunks to test the model on data it hasn't seen.
- **OOF (Out-Of-Fold)** — predictions on held-out chunks during cross-validation. Your honest performance estimate.
- **Ensemble** — averaging predictions from multiple models to get a more stable result.
- **Epoch** — one full pass through the training data.
- **F1** — harmonic mean of precision and recall. The metric we optimize.
- **Precision** — of the things I predicted, what fraction were right?
- **Recall** — of the things I should have predicted, what fraction did I get?
- **Checkpoint** — a saved snapshot of the model's weights at some point during training.
