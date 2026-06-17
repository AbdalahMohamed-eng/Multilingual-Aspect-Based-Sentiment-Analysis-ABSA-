<div align="center">
  <h1>🌍 Multilingual Aspect-Based Sentiment Analysis (ABSA)</h1>
  <p><i>A Production-Grade NLP Pipeline for Code-Switched Arabic & English Reviews</i></p>
</div>

---

## 📌 Project Overview
Customer reviews in the MENA region frequently blend Arabic and English (code-switching). Traditional Arabic-only Natural Language Processing (NLP) models fail to extract meaningful insights from these mixed-language inputs. 

This repository contains a robust, **Aspect-Based Sentiment Analysis (ABSA)** pipeline designed to solve this problem. Given a customer review, the system accurately detects:
1. **Aspects**: Specific topics mentioned (e.g., *Food, Service, Ambiance, App Experience, Price*).
2. **Sentiment Polarity**: The sentiment associated with each specific aspect (*Positive, Negative, Neutral*).

> **Example Input:** "الأكل ممتاز لكن الـ service كانت slow جداً." (The food was excellent but the service was very slow.)
> **Output:** `{"food": "positive", "service": "negative"}`

---

## 🏗️ System Architecture

To handle the complexity of multi-label aspect detection and multi-class sentiment classification, we developed a **Dual-Head Transformer Architecture**.

### 1. The Backbone: `XLM-RoBERTa-large`
* **The "Language Trap":** 98% of our training data was Arabic, but the test data contained 34% English. Instead of overfitting with an Arabic-only model like MARBERT, we utilized the 560M-parameter `XLM-RoBERTa-large`. This provided powerful zero-shot cross-lingual transfer, allowing the model to flawlessly understand English test queries despite being trained primarily on Arabic.

### 2. Dual Classification Heads
Instead of forcing a single layer to do everything, the semantic embeddings are split into two specialized heads:
* **Aspect Head (Multi-Label):** A Sigmoid activation head trained with Binary Cross-Entropy (BCE) loss to identify the presence of up to 9 distinct aspects.
* **Sentiment Head (Multi-Class):** A Softmax head trained with Masked Cross-Entropy (CE) loss to classify the polarity of the identified aspects.

### 3. Contextual Metadata Injection
Customer metadata significantly influences sentiment. We prepended structured metadata directly into the tokenized input string to provide the attention mechanism with strong prior context:
```text
[STARS=5] [CAT=Restaurant] [PLAT=Google_Maps] The food was amazing!
```

---

## ⚙️ Overcoming Technical Challenges

### 📉 1. Severe Class Imbalance (Mathematical Weighting)
Certain aspects (like *'cleanliness'* or *'delivery'*) appeared in less than 2% of the training dataset. Standard unweighted models completely ignored these rare classes to artificially boost aggregate accuracy.
* **Solution:** We implemented dynamic class weighting in our loss function (`pos_weight = sqrt((N - pos) / pos)`). This heavily penalized the model for missing rare classes, ensuring uniform recall across all 9 aspects.

### 🎛️ 2. Dynamic Threshold Tuning (Coordinate Ascent)
A static `0.5` probability threshold is suboptimal for imbalanced data. An aspect with 30% prevalence requires a different confidence threshold than an aspect with 1% prevalence.
* **Solution:** We implemented a Coordinate Ascent algorithm over 10 Out-Of-Fold (OOF) cross-validation models. The algorithm dynamically tuned and locked in the optimal probability threshold for each individual aspect independently, maximizing the global F1 score.

### 🛡️ 3. Smart Fallback Logic
In rare cases where the model confidence for all aspects fell below the tuned thresholds, returning an empty array would guarantee an F1 score of 0.
* **Solution:** We built an oracle fallback mechanism. If no threshold is met, the system falls back to the aspect with the absolute `argmax` probability and utilizes the user's star-rating metadata to intelligently infer the sentiment.

### ⚡ 4. Deployment Efficiency (Quantization & Memory Optimization)
Deploying a 560M-parameter model like `XLM-RoBERTa-large` in a production environment is computationally expensive and memory-intensive.
* **Solution:** To achieve low-latency inference without sacrificing our high F1 score, we optimized the model pipeline using **Mixed Precision (FP16)** and prepared the architecture for **INT8 Quantization**. This drastically reduces the VRAM footprint and speeds up token generation, making the heavy dual-head architecture viable for real-time customer feedback streams.

---

## 📊 Evaluation & Performance
The model was rigorously validated using a **10-Model Ensemble (5 Folds × 2 Random Seeds)** to ensure generalizability and eliminate variance.

* **Primary Metric:** Tuple Micro-F1 (Both Aspect + Sentiment must match)
* **Out-Of-Fold (OOF) F1 Score:** `0.7916`
* **Aspect-Only F1 Score:** `0.8502`
* **Sentiment Accuracy (Given Correct Aspect):** `0.9311`

---

## 🚀 Repository Structure
* `train.py`: The complete PyTorch training pipeline, including dataset formulation, weighted loss functions, and OOF evaluation.
* `submission.json`: The standard output JSON format generated by the model for evaluation.
* `DeepX_ABSA_Walkthrough.md`: A highly detailed, step-by-step technical diary of the iterative modeling process (V1 -> V2.3) and architectural decisions.

## 💻 Tech Stack
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![HuggingFace](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)