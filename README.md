

<div align="center">
  <h1>🛡️ VeriDex</h1>
  <h3>A Hybrid Multi-Modal Credibility Assessment System</h3>
  <p>Fake News Detection & Deepfake Image Forensics</p>
  
  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
  [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/rex177/VeriDex)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

**Author:** Rakesh Kumar Raut

---

## 📖 Abstract & Overview

AI-generated misinformation has exploded, and automated fact-verification systems are struggling to keep up. Modern Large Language Models (LLMs) churn out text that is almost impossible to tell apart from real language, while diffusion-based image synthesis creates deepfakes so realistic they make false claims look credible. These combined threats create a crucial "Zero-Day window" right after a big event, where independent fact-checks are missing, and evidence-based systems are helpless.

**VeriDex (Verified Deception Index)** is a Hybrid Multi-Modal Credibility Assessment System designed to tackle text from LLMs, AI-generated images, and the Zero-Day problem in one principled and transparent package. 

VeriDex runs three independently trained deep-learning pipelines and brings their results together through a mathematically derived **Hybrid Resolution Engine**. Tested on a custom 500-claim adversarial benchmark, VeriDex achieves state-of-the-art results, significantly outperforming text-only and retrieval-only baselines.

---

## 🏗️ Core Architecture (The Three Pipelines)

VeriDex takes a modular approach, with three parallel pipelines feeding into a central Hybrid Resolution Engine. This complementary redundancy ensures that each pipeline thrives exactly where the others fall short.

### 1. Linguistic Deception Analyzer (HierFND)
Catches linguistic deception based on writing style and text artifacts.
* **Model:** Fine-tuned `RoBERTa-base` sequence classifier.
* **Training Data:** LIAR, ISOT, and WELFake datasets.
* **Output:** A Linguistic Risk Score ($R_{ling}$) reflecting the probability of the text being machine-generated or stylistically deceptive.

### 2. Retrieval-Augmented Stance Aggregator (StanceFormer)
Verifies claims against actual evidence via real-time web retrieval.
* **Model:** Fine-tuned `DeBERTa-v3-base` Natural Language Inference (NLI) model.
* **Mechanism:** The claim triggers a web search API, pulling the top $k=10$ articles. 
* **Domain-Credibility RAG:** Each article’s domain gets mapped to a Media Bias/Fact Check (MBFC) credibility multiplier (1.5 for high-credibility, 1.0 for neutral, 0.2 for questionable/satire).
* **Output:** An Evidence Consensus Score ($E_{pro}$) indicating the factual support for the claim.

### 3. AI Image Forensics (CRAFT)
Pinpoints AI-generated images (deepfakes).
* **Architecture:** Combines a CLIP ViT-B/32 semantic branch with LoRA adapters and a 2D FFT-based **FrequencyBranch**.
* **Mechanism:** Merges 512-dimensional semantic features with 128-dimensional spectral features to catch checkerboard artifacts from GANs and diffusion models.
* **Output:** Image forensic classification (AI-Generated vs. Authentic) powered by a two-layer MLP head.

---

## ⚙️ Hybrid Resolution Engine

The core blend mixes linguistic risk with evidence-based risk using an empirically derived blending weight ($\alpha^* = 0.40$):

$$R_{total} = (R_{ling} \times 0.40) + ((1 - E_{pro}) \times 0.60)$$

This 40/60 split keeps pure text classifiers from missing LLM-generated fakes, while also countering the failures of retrieval-only approaches during the Zero-Day window.

### Fail-Safe Override Mechanisms
Two formal override rules ensure system robustness:
1. **Explicit Debunk Override:** If a highly credible article (weight $\ge 1.4$) strongly refutes the claim and uses terms like `"fact check"`, `"debunked"`, or `"false"`, the system forces $R_{total} \leftarrow \max(R_{total}, 0.90)$. This stops a clever lie from outweighing legitimate fact-checking.
2. **Image Forensics Override:** If an image is flagged as AI-Generated with $\ge 85\%$ confidence, the system forces $R_{total} \leftarrow \max(R_{total}, 0.85)$.

---

## 📊 Empirical Results & Benchmarks

VeriDex was rigorously evaluated on a custom **500-claim adversarial benchmark** encompassing Standard Fake News, Standard Real News, Zero-Day Claims, and LLM-Generated Fakes.

### Integrated System Performance
| Metric | Score | Category Accuracy Breakdown |
| :--- | :--- | :--- |
| **Accuracy** | 94.80% | Standard Fake: 97.5% |
| **AUC-ROC** | 0.965 | Standard Real: 98.0% |
| **Precision (Fake)** | 0.9751 | Zero-Day: 89.3% |
| **Recall (Fake)** | 0.9352 | LLM-Generated: 86.7% |
| **Macro-F1** | 0.9468 | |

> 🏆 **Comparison to State-of-the-Art (SOTA):** VeriDex beats FakeBERT by **+4.8 pp**, VeraCT Scan by **+3.3 pp**, and classical TF-IDF+SVM approaches by **+12.8 pp**.

### Image Forensics (CRAFT) Performance
Tested on a rigorous cross-generator protocol against 50,000 images (DALL-E 2, Midjourney v5, ProGAN, StyleGAN).
* **In-Distribution Accuracy:** 91.5%
* **Cross-Generator Accuracy:** 84.3% (Outperforming GenDet CVPR 2024 by 4.9 pp)
* **AUC-ROC:** 0.921

---

## 🌟 Novel Contributions

1. **Empirical 40/60 Weighting:** The first systematic grid search (101 configurations) over blending weights for linguistic and evidence information, backed by a bootstrapped 95% confidence interval [0.340, 0.462].
2. **Domain-Credibility RAG:** MBFC credibility multipliers act as continuous weights in evidence aggregation, boosting accuracy by +1.8 pp and blocking echo-chamber manipulation.
3. **Explicit Debunk Override:** A formal fail-safe ensuring high-credibility explicit fact-checks cannot be diluted by text model confidence (+1.4 pp accuracy).
4. **CLIP + LoRA + FrequencyBranch:** A novel parameter-efficient deepfake detector that slashes generalization degradation by 32% compared to CLIP-only setups.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (Highly Recommended for inference speed)

### 1. Clone & Install
```bash
git clone https://github.com/RakeshRautDev/VeriDex.git
cd VeriDex
pip install -r requirements.txt
```

### 2. Environment Variables
You need a Tavily API key for the Retrieval-Augmented web search.
Copy `.env.example` to `.env` and configure your API keys.
```bash
cp .env.example .env
# Edit .env to include: TAVILY_API_KEY=your_key_here
```

### 3. Automatic Model Weights Setup
✨ **New Feature:** You do not need to manually download heavy model binaries anymore! 

When you run `app.py`, the system will automatically connect to the `rex177/VeriDex-Weights` repository on Hugging Face Hub and dynamically download the fine-tuned RoBERTa, DeBERTa, and Image Forensics models to your local `models/` directory.

### 4. Run the Gradio Application
```bash
python app.py
```
Open the provided local URL (typically `http://127.0.0.1:7860`) in your browser to access the VeriDex verification dashboard.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
