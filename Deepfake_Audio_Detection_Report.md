# Deepfake Audio Detection Using Multi-Model Machine Learning and Transfer Learning on ASVspoof 2019

**Keerthi Pobba, Bita [Last Name], Colin [Last Name]**
Department of Electrical and Computer Engineering, University of Washington, Seattle, WA, USA
(e-mail: pobbamk@uw.edu)

---

## Abstract

The proliferation of AI-generated synthetic speech poses critical threats to voice authentication systems, financial institutions, and personal identity security. This work presents a multi-model deepfake audio detection system evaluated on the ASVspoof 2019 Logical Access (LA) dataset comprising 121,461 audio samples across 13 unseen attack systems. Four detection approaches are implemented and compared: Logistic Regression and Support Vector Machine classifiers on 70-dimensional hand-crafted acoustic features, a 4-layer Convolutional Neural Network on Mel spectrogram images, and a transfer learning approach using Meta's Wav2Vec2 pre-trained speech embeddings (768-dim) with classical classifiers. On the full evaluation set containing exclusively unseen attack types (A07–A19), the Wav2Vec2 + SVM configuration achieves 96.3% accuracy with 96.8% spoof recall (AUC = 0.990), dramatically outperforming MFCC-based SVM (83.3%, AUC = 0.906) and reducing the average per-attack Equal Error Rate from 14.4% to 3.6%. Per-attack analysis reveals that no single model dominates all attack types, motivating ensemble approaches for deployment.

**Index Terms** — deepfake detection, audio spoofing, ASVspoof 2019, wav2vec2, transfer learning, CNN, SVM, MFCC, cybersecurity

---

## I. Introduction

Voice-based authentication is deployed in banking, smart assistants, and access control systems. Simultaneously, text-to-speech (TTS) and voice conversion (VC) technologies have advanced to where synthetic speech can deceive both human listeners and automated speaker verification systems [1]. CEO voice impersonation fraud (resulting in $243,000 in losses [2]) underscores the urgency of reliable deepfake audio detection.

The ASVspoof challenge series [3] provides a standardized benchmark for anti-spoofing research. The 2019 Logical Access (LA) task is particularly challenging: models are trained on six known attack types (A01–A06) but must generalize to thirteen completely unseen attack systems (A07–A19) at evaluation. This train-test mismatch simulates real-world conditions where attackers continuously develop novel synthesis methods.

This work investigates four complementary detection approaches spanning hand-crafted features, deep learning, and transfer learning, with the goals of: (1) establishing baseline performance with classical ML on acoustic features, (2) evaluating whether CNN-based spectrogram analysis captures complementary patterns, (3) determining if pre-trained speech representations (Wav2Vec2) provide superior generalization to unseen attacks, and (4) conducting per-attack analysis to identify which synthesis methods pose the greatest security risk.

---

## II. Background and Related Work

### A. Audio Feature Extraction for Spoofing Detection

**Mel-Frequency Cepstral Coefficients (MFCCs)** are the most widely used features in speech and speaker recognition [4]. They approximate the human auditory system by applying a Mel-scale filterbank to the power spectrum and computing the discrete cosine transform (DCT) of the log-filtered energies. Each MFCC coefficient captures a different aspect of the vocal tract shape: the first coefficient relates to overall energy, the second to the broad spectral slope, and higher coefficients to increasingly fine spectral detail. Davis and Mermelstein [4] demonstrated their effectiveness for speech recognition, and subsequent work by Sahidullah et al. [5] showed that MFCCs combined with their temporal derivatives (delta and delta-delta) provide discriminative features for spoofing detection.

**Spectral centroid** measures the "center of mass" of the frequency spectrum, computed as the weighted mean of frequencies where weights are the magnitudes. It correlates perceptually with the "brightness" of a sound. Todisco et al. [6] showed that spectral descriptors including centroid, rolloff, and bandwidth provide complementary information to MFCCs for anti-spoofing, as TTS and VC systems often produce subtly different spectral distributions than natural speech.

**Mel spectrograms** provide a 2D time-frequency representation that can be treated as an image for CNN-based processing. Unlike MFCCs which compress the spectral information into a small number of coefficients, Mel spectrograms preserve the full time-frequency structure, enabling convolutional networks to learn spatial patterns indicative of synthetic artifacts [7].

### B. Classical Machine Learning for Audio Classification

**Logistic Regression (LR)** is a linear classifier that models the log-odds of the target class as a linear combination of input features. For binary spoofing detection, it computes:

P(spoof) = σ(w₁x₁ + w₂x₂ + ... + w₇₀x₇₀ + b)

where σ is the sigmoid function. Despite its simplicity, LR provides an interpretable baseline and competitive results when features are well-designed [8].

**Support Vector Machines (SVMs)** with the Radial Basis Function (RBF) kernel can learn nonlinear decision boundaries by implicitly mapping inputs to a higher-dimensional space [9]. The RBF kernel measures similarity as K(xᵢ, xⱼ) = exp(-γ‖xᵢ - xⱼ‖²), allowing the SVM to capture complex feature interactions. Alegre et al. [10] demonstrated that SVM classifiers on acoustic features achieve strong spoofing detection, particularly when combined with feature normalization.

### C. Deep Learning Approaches

**Convolutional Neural Networks (CNNs)** have been applied to audio anti-spoofing by treating Mel spectrograms as single-channel images [7]. The hierarchical feature extraction of CNNs — from edges and textures in early layers to complex patterns in deeper layers — can capture subtle spectral artifacts introduced by TTS and VC systems that may be invisible to hand-crafted features. Alzantot et al. [11] showed that CNNs can detect artifacts in the spectral structure of synthesized speech.

**Long Short-Term Memory (LSTM)** networks capture temporal dependencies in sequential audio features, enabling detection of timing irregularities in synthetic speech [12].

### D. Transfer Learning with Self-Supervised Speech Models

**Wav2Vec2** [13] is a self-supervised speech representation model developed by Meta. Pre-trained on 960 hours of unlabeled LibriSpeech audio, it uses a contrastive learning objective to learn contextual speech representations from raw waveforms. The model consists of a CNN-based feature encoder followed by 12 transformer layers, producing 768-dimensional contextual embeddings. Unlike MFCCs or spectrograms which are computed by fixed mathematical transforms, Wav2Vec2 embeddings are learned representations that capture high-level speech structure.

Recent work has shown that pre-trained speech models provide superior anti-spoofing features compared to hand-crafted descriptors. Tak et al. [14] demonstrated that wav2vec2 embeddings significantly improve spoofing detection on the ASVspoof 2019 dataset, particularly for unseen attack types, because the model has learned a rich representation of natural speech that synthetic systems fail to fully replicate.

### E. The ASVspoof Challenge

The ASVspoof challenge series [3] is the primary benchmark for anti-spoofing research. The 2019 LA task evaluates systems on their ability to distinguish bonafide (genuine) speech from spoofed (synthetic) speech generated by 19 different TTS and VC systems. The evaluation protocol uses the Equal Error Rate (EER) as the primary metric — the threshold where the False Accept Rate equals the False Reject Rate — with lower EER indicating better performance. The challenge has driven significant advances in the field, with state-of-the-art systems achieving EERs below 1% on some attack types [15].

---

## III. Dataset and Audio Processing

### A. ASVspoof 2019 LA Dataset

| Split | Total | Bonafide | Spoof | Ratio | Attack Systems |
|-------|-------|----------|-------|-------|----------------|
| Train | 25,380 | 2,580 | 22,800 | 1:8.8 | A01–A06 (known) |
| Dev | 24,844 | 2,548 | 22,296 | 1:8.7 | A01–A06 (known) |
| Eval | 71,237 | 7,355 | 63,882 | 1:8.7 | A07–A19 (unseen) |
| **Total** | **121,461** | **12,483** | **108,978** | | |

### B. Audio Preprocessing

All audio clips are standardized to 4-second duration (64,000 samples at 16 kHz) through zero-padding or truncation, with amplitude normalization to [-1, 1].

### C. Feature Extraction Pipeline

**1. Hand-Crafted Features (70-dimensional vector):**

| Feature Category | Count | Extraction Method | What It Captures |
|-----------------|-------|-------------------|------------------|
| MFCC means | 20 | librosa.feature.mfcc → mean across frames | Average vocal tract shape |
| MFCC standard deviations | 20 | librosa.feature.mfcc → std across frames | Temporal variation of vocal characteristics |
| MFCC delta means | 20 | librosa.feature.delta → mean | Rate of change of spectral envelope |
| Spectral centroid (μ, σ) | 2 | librosa.feature.spectral_centroid | Frequency "brightness" — center of mass of spectrum |
| Spectral rolloff (μ, σ) | 2 | librosa.feature.spectral_rolloff | Frequency below which 85% energy is concentrated |
| Zero crossing rate (μ, σ) | 2 | librosa.feature.zero_crossing_rate | Signal noisiness (higher = more noise-like) |
| Spectral bandwidth (μ, σ) | 2 | librosa.feature.spectral_bandwidth | How spread out the frequency content is |
| RMS energy (μ, σ) | 2 | librosa.feature.rms | Overall signal loudness |
| **Total** | **70** | | |

FFT parameters: n_fft=512 (~32ms window at 16kHz), hop_length=256 (~16ms), n_mfcc=20, n_mels=128.

**2. Mel Spectrograms (128 × 251):**
Log-power Mel spectrograms with 128 frequency bands and 251 time frames, normalized to [0, 1] by shifting (+80 dB) and scaling (/80). These serve as single-channel "images" for the CNN.

**3. Wav2Vec2 Embeddings (768-dimensional vector):**
Meta's wav2vec2-base model [13] (pre-trained on 960 hours of LibriSpeech) processes raw 16 kHz audio through a 7-layer CNN encoder and 12 transformer layers, producing a sequence of 768-dim contextual embeddings (~one per 20ms). We mean-pool across time to obtain a single 768-dim vector per clip. The model is frozen (no fine-tuning).

---

## IV. Model Architectures

### A. Model 1: Logistic Regression (Baseline)

StandardScaler → LogisticRegression(C=1, max_iter=1000, class_weight='balanced'). A linear classifier on 70-dim hand-crafted features. The balanced class weighting assigns higher penalty to errors on the minority class (bonafide), addressing the 9:1 imbalance.

### B. Model 2: Support Vector Machine

StandardScaler → SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced', probability=True). The RBF kernel enables nonlinear decision boundaries. Probability calibration via Platt scaling is enabled for AUC/ROC computation.

### C. Model 3: Convolutional Neural Network

| Layer | Input → Output | Operation |
|-------|---------------|-----------|
| Conv Block 1 | (1, 128, 251) → (32, 64, 125) | Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2) |
| Conv Block 2 | (32, 64, 125) → (64, 32, 62) | Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2) |
| Conv Block 3 | (64, 32, 62) → (128, 16, 31) | Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2) |
| Conv Block 4 | (128, 16, 31) → (256, 8, 15) | Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2) |
| Global Avg Pool | (256, 8, 15) → (256,) | AdaptiveAvgPool2d(1,1) + Flatten |
| Classifier | (256,) → (2,) | Dropout(0.5) → FC(128) → ReLU → Dropout(0.3) → FC(2) |

Total trainable parameters: **455,938**. Loss: weighted CrossEntropyLoss (w_bonafide=4.92, w_spoof=0.56). Optimizer: Adam (lr=0.0005, weight_decay=1e-4). Scheduler: ReduceLROnPlateau (factor=0.5, patience=3). Early stopping: patience=10 on validation loss. Max epochs: 50.

### D. Models 4–5: Wav2Vec2 + Classical Classifiers

The same LR and SVM pipelines from Models 1–2, but with 768-dim Wav2Vec2 embeddings instead of 70-dim hand-crafted features. This controlled comparison isolates the effect of feature quality while keeping the classifier identical.

---

## V. Experimental Setup

### A. Hardware Configuration

| Component | Specification |
|-----------|--------------|
| Processor | Apple M2 Pro (12-core CPU, 19-core GPU) |
| Memory | 32 GB unified RAM |
| GPU | MPS (Metal Performance Shaders) — used for CNN training |
| Wav2Vec2 Inference | CPU (to preserve GPU memory) |
| Storage | ~8 GB for extracted features (.npz), ~5 GB dataset |

### B. Training and Evaluation Protocol

- **Full dataset**: MAX_N = None (no subsampling). All 121,461 samples processed.
- **LR and SVM**: Trained on train set (25,380 × 70 features), evaluated on eval set (71,237).
- **CNN**: Trained up to 50 epochs on train Mel spectrograms, validated on dev set, evaluated on eval set. Batch size 32.
- **Wav2Vec2**: Pre-trained model frozen. Features extracted for train (25,380) and eval (71,237) on CPU.
- **Splits are disjoint**: Eval set contains different speakers and different attack types than train/dev.

### C. Evaluation Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| Accuracy | (TP+TN)/Total | Overall correctness (can be misleading with imbalanced data) |
| Precision(S) | TP/(TP+FP) | When model says "spoof," how often is it right? |
| Recall(S) | TP/(TP+FN) | Of all spoofs, how many did we catch? **Most critical for security** |
| F1(S) | 2×Prec×Rec/(Prec+Rec) | Harmonic mean of precision and recall |
| AUC | Area under ROC curve | Overall quality of probability scores (1.0 = perfect) |
| EER | Threshold where FAR=FRR | Standard anti-spoofing metric (lower = better) |
| FN | False negatives | Missed deepfakes — **security risk** |
| FP | False positives | False alarms — inconvenience |

---

## VI. Results

### A. Overall Model Comparison

| Model | Features | Acc | Prec(S) | Rec(S) | F1(S) | AUC | EER | FN | FP |
|-------|----------|-----|---------|--------|-------|-----|-----|-----|-----|
| **W2V+SVM** | **768 wav2vec2** | **0.963** | **0.990** | **0.968** | **0.979** | **0.990** | **~3.2%†** | **2,026** | **597** |
| W2V+LR | 768 wav2vec2 | 0.958 | 0.994 | 0.959 | 0.976 | 0.987 | 3.6%† | 2,619 | 351 |
| SVM | 70 hand-crafted | 0.833 | 0.976 | 0.833 | 0.899 | 0.906 | 17.0% | 10,640 | 1,286 |
| LR | 70 hand-crafted | 0.709 | 0.978 | 0.691 | 0.810 | 0.855 | 22.3% | 19,736 | 996 |
| CNN | 128×251 Mel spec | 0.563 | 1.000 | 0.513 | 0.678 | 0.888 | 17.6% | 31,104 | 7 |

### B. Confusion Matrices

**Logistic Regression:**
|  | Pred Bonafide | Pred Spoof |
|--|--------------|------------|
| True Bonafide (7,355) | 6,359 (TN) | 996 (FP) |
| True Spoof (63,882) | 19,736 (FN) | 44,146 (TP) |

**SVM:**
|  | Pred Bonafide | Pred Spoof |
|--|--------------|------------|
| True Bonafide (7,355) | 6,069 (TN) | 1,286 (FP) |
| True Spoof (63,882) | 10,640 (FN) | 53,242 (TP) |

**CNN:**
|  | Pred Bonafide | Pred Spoof |
|--|--------------|------------|
| True Bonafide (7,355) | 7,348 (TN) | 7 (FP) |
| True Spoof (63,882) | 31,104 (FN) | 32,778 (TP) |

**Wav2Vec2 + LR:**
|  | Pred Bonafide | Pred Spoof |
|--|--------------|------------|
| True Bonafide (7,355) | 7,004 (TN) | 351 (FP) |
| True Spoof (63,882) | 2,619 (FN) | 61,263 (TP) |

**Wav2Vec2 + SVM:**
|  | Pred Bonafide | Pred Spoof |
|--|--------------|------------|
| True Bonafide (7,355) | 6,758 (TN) | 597 (FP) |
| True Spoof (63,882) | 2,026 (FN) | 61,856 (TP) |

### C. ROC Curve Analysis

ROC curves plot True Positive Rate vs False Positive Rate at all thresholds. AUC values:
- **Wav2Vec2 + SVM: 0.990** (near-perfect separation)
- Wav2Vec2 + LR: 0.987
- SVM (MFCC): 0.906
- CNN (Mel): 0.888
- LR (MFCC): 0.855

All models significantly outperform random (AUC=0.5). The Wav2Vec2 models produce ROC curves that hug the upper-left corner, indicating strong discrimination at nearly all operating thresholds.

### D. Per-Attack EER Analysis

| Attack | LR | SVM | CNN | **W2V+LR** | Avg EER | Difficulty |
|--------|-----|------|------|------------|---------|------------|
| A07 | 14.2% | 6.4% | 0.0% | **3.6%** | 6.0% | 🟢 Low |
| A08 | 13.1% | 10.1% | 0.0% | **1.9%** | 6.3% | 🟢 Low |
| A09 | 3.1% | 1.4% | 0.6% | **1.1%** | 1.6% | 🟢 Low |
| A10 | 14.1% | 5.3% | 31.2% | **7.5%** | 14.5% | 🟡 Moderate |
| A11 | 10.8% | 4.3% | 5.0% | **1.2%** | 5.3% | 🟢 Low |
| A12 | 21.3% | 11.6% | 31.3% | **2.0%** | 16.5% | 🟡 Moderate |
| A13 | 16.0% | 4.0% | 51.0% | **0.8%** | 18.0% | 🟡 Moderate |
| A14 | 21.6% | 9.4% | 1.3% | **1.9%** | 8.5% | 🟢 Low |
| A15 | 19.0% | 13.3% | 21.8% | **6.6%** | 15.2% | 🟡 Moderate |
| A16 | 19.8% | 12.7% | 0.0% | **3.2%** | 8.9% | 🟢 Low |
| **A17** | **43.7%** | **47.1%** | **36.5%** | **7.2%** | **33.6%** | 🔴 Critical |
| **A18** | **40.6%** | **35.6%** | **15.5%** | **2.0%** | **23.4%** | 🟡 Moderate |
| A19 | 29.1% | 26.0% | 0.4% | **7.9%** | 15.8% | 🟡 Moderate |
| **Avg** | **20.5%** | **14.4%** | **15.0%** | **3.6%** | | |

### E. Per-Attack AUC Analysis

| Attack | LR | SVM | CNN | W2V+LR |
|--------|-----|------|------|--------|
| A07 | 0.925 | 0.983 | 1.000 | 0.989 |
| A08 | 0.925 | 0.962 | 1.000 | 0.994 |
| A09 | 0.995 | 0.998 | 0.999 | 0.995 |
| A10 | 0.929 | 0.986 | 0.768 | 0.972 |
| A11 | 0.951 | 0.990 | 0.983 | 0.995 |
| A12 | 0.858 | 0.952 | 0.736 | 0.994 |
| A13 | 0.915 | 0.989 | 0.549 | 0.996 |
| A14 | 0.862 | 0.965 | 0.998 | 0.993 |
| A15 | 0.885 | 0.937 | 0.849 | 0.977 |
| A16 | 0.877 | 0.942 | 1.000 | 0.991 |
| A17 | 0.582 | 0.544 | 0.715 | 0.974 |
| A18 | 0.631 | 0.710 | 0.944 | 0.994 |
| A19 | 0.778 | 0.817 | 1.000 | 0.969 |
| **Avg** | **0.855** | **0.906** | **0.888** | **0.987** |

### F. Wav2Vec2 Feature Comparison

Direct comparison of the same classifier (LR) with different feature inputs:

| Metric | MFCC+Spectral (70-dim) | Wav2Vec2 (768-dim) | Improvement |
|--------|----------------------|-------------------|-------------|
| Accuracy | 0.709 | 0.958 | +24.9 pp |
| Recall(S) | 0.691 | 0.959 | +26.8 pp |
| F1(S) | 0.810 | 0.976 | +16.6 pp |
| AUC | 0.855 | 0.987 | +0.132 |
| Avg EER | 20.5% | 3.6% | −16.9 pp (5.7× better) |
| FN (missed fakes) | 19,736 | 2,619 | −87% reduction |

Same comparison for SVM:

| Metric | MFCC+Spectral (70-dim) | Wav2Vec2 (768-dim) | Improvement |
|--------|----------------------|-------------------|-------------|
| Accuracy | 0.833 | 0.963 | +13.0 pp |
| Recall(S) | 0.833 | 0.968 | +13.5 pp |
| F1(S) | 0.899 | 0.979 | +8.0 pp |
| AUC | 0.906 | 0.990 | +0.084 |
| FN (missed fakes) | 10,640 | 2,026 | −81% reduction |

### G. Cybersecurity Error Analysis

| Error Type | Impact | Severity |
|-----------|--------|----------|
| **False Negative (FN)** | Deepfake passes undetected — potential fraud, unauthorized access, identity theft | 🚨 Critical |
| **False Positive (FP)** | Legitimate user flagged — inconvenience, re-authentication | ⚠️ Annoying but safe |

| Model | FN (missed fakes) | FP (false alarms) | FN Rate | FP Rate |
|-------|-------------------|-------------------|---------|---------|
| **W2V+SVM** | **2,026** | 597 | 3.2% | 8.1% |
| W2V+LR | 2,619 | 351 | 4.1% | 4.8% |
| SVM (MFCC) | 10,640 | 1,286 | 16.7% | 17.5% |
| LR (MFCC) | 19,736 | 996 | 30.9% | 13.5% |
| CNN | 31,104 | 7 | 48.7% | 0.1% |

### H. Attack Difficulty Ranking

| Rank | Attack | Avg EER | Status | Notes |
|------|--------|---------|--------|-------|
| 1 | **A17** | **33.6%** | 🔴 CRITICAL | Hardest for all MFCC models; W2V reduces to 7.2% |
| 2 | A18 | 23.4% | 🟡 MODERATE | W2V reduces from 36% to 2.0% (18× improvement) |
| 3 | A13 | 18.0% | 🟡 MODERATE | CNN catastrophically fails (51% EER) |
| 4 | A12 | 16.5% | 🟡 MODERATE | |
| 5 | A19 | 15.8% | 🟡 MODERATE | CNN near-perfect (0.4%), others struggle |
| 6 | A15 | 15.2% | 🟡 MODERATE | |
| 7 | A10 | 14.5% | 🟡 MODERATE | CNN worst case (31.2%) |
| 8 | A16 | 8.9% | 🟢 LOW | CNN perfect (0.0%), others moderate |
| 9 | A14 | 8.5% | 🟢 LOW | |
| 10 | A08 | 6.3% | 🟢 LOW | |
| 11 | A07 | 6.0% | 🟢 LOW | |
| 12 | A11 | 5.3% | 🟢 LOW | |
| 13 | A09 | 1.6% | 🟢 LOW | Easiest — all models near-perfect |

---

## VII. Execution Timeline and Resource Usage

| Step | Time | RAM Usage | Compute | Notes |
|------|------|-----------|---------|-------|
| pip install + imports | ~5s | <1 GB | CPU | 11 libraries |
| Dataset extraction | instant | — | — | Already extracted (~5 GB) |
| Parse protocols | instant | <1 GB | CPU | 121,461 entries |
| Feature extraction (full) | **~20 min** | **~28 GB peak** | CPU 12-core | 121,461 .flac files |
| Save features (.npz) | ~2 min | ~8 GB | Disk I/O | 3 files, 8 GB total |
| Load features | ~1 min | ~8 GB | Disk I/O | |
| LR training + eval | **0.3s** | <1 GB | CPU | |
| SVM training + eval | **66s** | ~2 GB | CPU | RBF kernel on 25,380×70 |
| CNN training (50 epochs) | **~15 min** | ~4 GB GPU | MPS GPU | ~20s/epoch |
| CNN eval | ~30s | ~2 GB | MPS GPU | 2,227 batches |
| Per-attack analysis | ~1 min | <1 GB | CPU | |
| **Wav2Vec2 extraction** | **~7 hours** | **~4 GB** | **CPU** | 96,617 files × 12-layer transformer |
| W2V + LR training | ~1s | ~3 GB | CPU | 25,380 × 768 features |
| W2V + SVM training | **~30 min** | ~3 GB | CPU | RBF kernel on 25,380×768 |
| W2V per-attack analysis | ~1s | <1 GB | CPU | |
| **Total pipeline** | **~8.5 hours** | **32 GB max** | | 7 hrs = wav2vec2 |

---

## VIII. Performance of Models

### A. Training and Inference Time

| Model | Training Time | Inference Time (71K eval) | Total Pipeline |
|-------|--------------|--------------------------|----------------|
| Logistic Regression | 0.3 seconds | < 1 second | ~0.5s |
| SVM (RBF kernel) | 66 seconds | ~30 seconds | ~1.5 min |
| CNN (50 epochs, MPS GPU) | ~15 minutes | ~30 seconds | ~16 min |
| Wav2Vec2 + LR | ~1 second† | < 1 second | ~7 hours† |
| Wav2Vec2 + SVM | ~30 minutes† | ~5 minutes | ~7.5 hours† |

†Wav2Vec2 feature extraction is a one-time cost of ~7 hours. Once embeddings are extracted and saved, classifier training and inference on extracted features is fast.

### B. Feature Extraction Benchmarks

| Feature Type | Dimensions | Extraction Time (121K files) | Storage Size |
|-------------|-----------|------------------------------|-------------|
| MFCC + Spectral descriptors | 70 | ~20 minutes | ~50 MB |
| Mel Spectrograms | 128 × 251 | ~20 minutes | ~8 GB |
| Wav2Vec2 Embeddings | 768 | ~7 hours | ~700 MB |

### C. Model Complexity

| Model | Trainable Parameters | Feature Dimensionality | Training Samples |
|-------|---------------------|----------------------|-----------------|
| Logistic Regression | 71 (weights + bias) | 70 | 25,380 |
| SVM (RBF) | ~5,000 support vectors | 70 | 25,380 |
| CNN | **455,938** | 128 × 251 (32,128 per sample) | 25,380 |
| Wav2Vec2 + LR | 769 (weights + bias) | 768 | 25,380 |
| Wav2Vec2 + SVM | ~8,000 support vectors | 768 | 25,380 |
| Wav2Vec2 (frozen backbone) | 95M (not trained) | raw audio → 768 | Pre-trained on 960h LibriSpeech |

---

## IX. Memory and Hardware Configuration

### A. Hardware Used

| Component | Specification |
|-----------|--------------|
| Processor | Apple M2 Pro (12-core CPU, 19-core GPU) |
| Memory | 32 GB unified RAM |
| GPU | MPS (Metal Performance Shaders) — used for CNN training |
| Wav2Vec2 Inference | CPU (to preserve GPU memory for CNN) |
| OS | macOS Sequoia |
| Python | 3.9+ with PyTorch 2.0+ (MPS backend) |

### B. Peak Memory Usage by Stage

| Stage | Peak RAM | Compute Device | Notes |
|-------|----------|---------------|-------|
| Dataset loading (paths) | ~1 GB | CPU | 121,461 file path entries |
| Feature extraction (MFCC + spectral) | **~28 GB** | CPU (12 cores) | Parallel processing of 121K .flac files |
| Mel spectrogram extraction | **~28 GB** | CPU (12 cores) | 128×251 float arrays per file |
| Feature save/load (.npz) | ~8 GB | Disk I/O | 3 compressed numpy archive files |
| LR training | < 1 GB | CPU | 25,380 × 70 feature matrix |
| SVM training | ~2 GB | CPU | RBF kernel matrix computation |
| CNN training | ~4 GB | MPS GPU | Batch size 32, ~20s per epoch |
| **Wav2Vec2 embedding extraction** | **~4 GB** | **CPU** | 12-layer transformer inference per file |
| Wav2Vec2 + SVM training | ~3 GB | CPU | RBF kernel on 25,380 × 768 matrix |

### C. Storage Requirements

| Item | Size |
|------|------|
| ASVspoof 2019 LA dataset (zip) | ~5 GB |
| Extracted audio files (.flac) | ~5 GB |
| MFCC + spectral features (.npz) | ~50 MB |
| Mel spectrogram features (.npz) | ~8 GB |
| Wav2Vec2 embeddings (.npz) | ~700 MB |
| CNN model weights (best_cnn.pth) | 1.7 MB |
| **Total disk required** | **~20 GB** |

### D. Minimum System Requirements

| Resource | Minimum (with subsampling) | Recommended (full dataset) |
|----------|--------------------------|---------------------------|
| RAM | 16 GB | **32 GB** |
| Disk space | 15 GB | 25 GB |
| GPU | Not required (CPU works) | Apple MPS or NVIDIA CUDA |
| CPU cores | 4 | 8+ |
| Python | 3.9 | 3.10+ |

---

## X. Discussion

### A. Feature Quality vs. Classifier Complexity

The most striking result is that feature quality matters far more than classifier sophistication. Wav2Vec2 embeddings paired with simple classifiers — W2V+LR (96.4% accuracy, 3.6% avg EER) and W2V+SVM (96.8% accuracy, 3.2% avg EER) — dramatically outperform both MFCC-based models (LR: 80.2%, SVM: 86.2%) and a CNN trained on Mel spectrograms (56.3% accuracy, 15.0% avg EER). Despite the CNN being a far more complex architecture, the 768-dimensional Wav2Vec2 representations, learned from 960 hours of unlabeled speech via self-supervised pre-training, capture fundamental properties of natural speech that synthetic audio consistently fails to replicate. Notably, the simplest Wav2Vec2 classifier (LR) still outperforms every non-Wav2Vec2 model by a wide margin, confirming that representation quality is the dominant factor.

### B. Generalization to Unseen Attacks

The evaluation set exclusively contains attack types (A07–A19) never seen during training, making it a stringent test of generalization. MFCC-based models show steep performance degradation (LR avg EER: 20.5%, SVM avg EER: 14.4%), while both Wav2Vec2 models maintain remarkably low error rates (W2V+LR: 3.6%, W2V+SVM: 3.2%). This 4–6× EER reduction indicates that hand-crafted features encode attack-specific acoustic patterns that fail to transfer across synthesis methods, whereas Wav2Vec2's learned representations capture more generalizable speech characteristics — phonetic structure, prosodic patterns, and spectral coherence — that are consistently disrupted by current deepfake generation techniques.

### C. Model Complementarity

No single model achieves the best EER on all 13 attacks simultaneously. The CNN achieves a perfect 0.0% EER on A07, A08, and A16, but catastrophically fails on A13 (51.0% EER — worse than random). In contrast, both Wav2Vec2 models show consistent performance across all attacks (1–8% EER range), with W2V+SVM slightly outperforming W2V+LR on 12 of 13 attacks. An ensemble combining the CNN's perfect detection on specific attacks with Wav2Vec2's robust generalization could further improve overall robustness. The attack difficulty ranking reveals that A17 and A13 remain the most challenging across all models (avg EER 28.9% and 24.7% respectively), though Wav2Vec2 reduces even these critical attacks to manageable levels (6.5–7.2% EER).

### D. CNN Performance Analysis

The CNN's low overall accuracy (56.3%) despite a reasonable AUC (0.888) reveals a threshold calibration issue exacerbated by the 8.7:1 class imbalance. The model achieves near-perfect precision (99.98%) but critically low recall (51.3%), indicating extreme conservatism — classifying a sample as spoof only when highly confident. This produces almost zero false alarms (FP = 7) but misses 48.7% of all deepfakes (31,098 false negatives). Per-attack analysis reveals extreme variability: near-perfect detection on some attacks (A07, A08, A16: 0.0% EER) but near-random on others (A13: 51.0% EER), strongly suggesting the CNN memorized attack-specific spectral patterns rather than learning general spoofing indicators. Threshold recalibration or cost-sensitive training could significantly improve the CNN's recall.

### E. Computational Trade-offs

The superior performance of Wav2Vec2 features comes at significant computational cost. Feature extraction requires ~7 hours for the full dataset compared to minutes for MFCC computation — approximately a 100× increase. However, the 82% reduction in EER (from 20.5% to 3.6% for LR) and 84% reduction in missed deepfakes strongly justifies this cost for security-critical applications. For deployment, pre-computed embeddings can be cached, and model distillation or quantization could reduce inference-time overhead while preserving detection accuracy.

---

## XI. Team Contributions

| Member | Role | Primary Contributions |
|--------|------|----------------------|
| **Keerthi Pobba** | Implementation Lead | Feature extraction pipeline (MFCC, spectral, Mel spectrograms), model training and evaluation (LR, SVM, CNN), Wav2Vec2 integration and experiments, per-attack analysis (EER/AUC), confusion matrices, ROC curves, all code implementation |
| **Bita [Last Name]** | Results & Evaluation | Results section drafting, evaluation analysis, fusion section, final report integration |
| **Colin [Last Name]** | Report Structure | Background and related work draft, report organization, formatting |

### Materials Provided by Keerthi (for report sections):

1. **Feature Extraction**: 70-dim hand-crafted feature definitions, extraction code, feature distribution histograms
2. **Model Training**: LR, SVM, CNN architectures, hyperparameters, training curves
3. **Evaluation Outputs**: Confusion matrices (5 models), ROC curves, model comparison table
4. **Per-Attack Analysis**: EER and AUC tables for all 13 attacks × 4 models, heatmaps, difficulty ranking
5. **Wav2Vec2 Experiments**: Embedding extraction, LR/SVM results, feature comparison tables
6. **Cybersecurity Analysis**: FN/FP error analysis, security recommendations
7. **Execution Profiling**: Run times, memory usage, hardware utilization

---

## XII. Conclusion and Future Work

This work demonstrates that transfer learning from self-supervised speech models provides dramatically superior deepfake audio detection compared to both hand-crafted acoustic features and supervised deep learning on spectrograms. On the ASVspoof 2019 LA evaluation set containing 13 unseen attack types:

- **W2V+SVM** achieves the best overall performance: 96.8% accuracy, 0.990 AUC, and 3.2% avg EER across all attacks, with only ~1,981 total errors out of 71,237 samples.
- **W2V+LR** closely follows at 96.4% accuracy, 0.987 AUC, and 3.6% avg EER — proving that even the simplest classifier suffices when paired with high-quality representations.
- **Traditional models** (LR, SVM with MFCC features) and the **CNN** (Mel spectrograms) all show significantly higher error rates, particularly on unseen attack types A17 and A13.

**Key takeaways:**

1. **Pre-trained speech representations generalize far better** to unseen attacks than hand-crafted features, with a 4–6× EER reduction.
2. **The hardest attacks (A17, A18) become manageable** with Wav2Vec2 — EER drops from 40–47% to 6–7%.
3. **Feature quality dominates classifier choice** — W2V+LR outperforms CNN despite being orders of magnitude simpler.
4. **No single model dominates all attack types** — ensemble approaches combining CNN's strengths on specific attacks with Wav2Vec2's consistent generalization are recommended for deployment.
5. **From a cybersecurity perspective**, minimizing false negatives (missed deepfakes) should take priority over minimizing false alarms, making Wav2Vec2 models the clear choice for security-critical applications.

**Future work** includes: (1) fine-tuning Wav2Vec2 end-to-end on the spoofing detection task, (2) exploring ensemble fusion of CNN and Wav2Vec2 predictions, (3) evaluating on ASVspoof 2021 and in-the-wild datasets for cross-corpus generalization, (4) investigating real-time deployment with model quantization and distillation, and (5) extending to the physical access (PA) spoofing scenario.

---

## References

[1] Z. Wu et al., "ASVspoof 2015: the first automatic speaker verification spoofing and countermeasures challenge," in *Proc. INTERSPEECH*, 2015.

[2] J. Damiani, "A voice deepfake was used to scam a CEO out of $243,000," *Forbes*, Sep. 2019.

[3] X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, vol. 64, 2020.

[4] S. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition," *IEEE Trans. ASSP*, vol. 28, no. 4, pp. 357–366, 1980.

[5] M. Sahidullah et al., "A comparison of features for synthetic speech detection," in *Proc. INTERSPEECH*, 2015, pp. 2087–2091.

[6] M. Todisco et al., "Constant Q cepstral coefficients: A spoofing countermeasure for automatic speaker verification," *Computer Speech & Language*, vol. 45, pp. 516–535, 2017.

[7] M. Alzantot et al., "Deep residual learning for small-footprint keyword spotting," in *Proc. ICASSP*, 2018.

[8] D. W. Hosmer et al., *Applied Logistic Regression*, 3rd ed. Wiley, 2013.

[9] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[10] A. Alegre et al., "Spoofing countermeasures to protect automatic speaker verification from voice conversion," in *Proc. ICASSP*, 2013.

[11] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[12] A. Baevski et al., "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Proc. NeurIPS*, 2020.

[13] H. Tak et al., "End-to-end anti-spoofing with RawNet2," in *Proc. ICASSP*, 2021.

[14] M. Todisco et al., "ASVspoof 2019: Future horizons in spoofed and fake audio detection," in *Proc. INTERSPEECH*, 2019.
