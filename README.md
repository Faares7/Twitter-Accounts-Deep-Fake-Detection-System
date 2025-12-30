# Twitter Accounts Deep Fake Detection System
## "The Shield Against Digital Deception"

**Samsung Innovation Campus (SIC) - AI Team 17**

## 🛡️ Overview

This project addresses the critical threat of deepfakes and synthetic media to information integrity. We have developed an intelligent system designed to detect fake accounts on social media (specifically X/Twitter) by analyzing both profile images and posted media.

The system distinguishes between real, AI-generated, stolen, or manipulated content to identify whether an account is likely Real, Fake, or Bot-controlled.

## 🚩 Problem Statement

* **Proliferation of Fakes:** Social media platforms are saturated with AI-generated faces, bot accounts, and impersonations.

* **Detection Difficulty:** Deepfakes are advancing rapidly, making detection difficult for human perception and traditional algorithms.

* **Visual Convincingness:** Forgery evidence is often hidden in high-frequency noise patterns and latent inconsistencies rather than obvious visual flaws.

## 🏗️ Technical Architecture

The solution utilizes a dual-model pipeline leveraging Transfer Learning to reduce computational costs while ensuring powerful feature extraction.

### 1. Face Authenticity Model (Real vs. Fake)

* **Backbone:** ResNet50 (Pre-trained on ImageNet).

* **Framework:** PyTorch Lightning & `timm` library for standardized interfaces.

* **Training Strategy:**
   * **Phase 1 (Head Adaptation):** Backbone frozen, LR=$10^{-4}$. Trains only the classification head.
   * **Phase 2 (Full Fine-Tuning):** Unfrozen backbone, LR=$10^{-5}$. Prevents "catastrophic forgetting" of generalized ImageNet features.

### 2. Profile Classification Model (Bot Detection)

* **Model:** `x-bot-profile-detection` (HuggingFace).

* **Architecture:** Built on top of SigLIP2 (`google/siglip2-base-patch16-224`).

* **Classes:** Classifies profiles into four categories: Bot, Cyborg, Real, Verified.

## 📊 Dataset

* **Source:** Collection of generated and real accounts on X (Twitter).

* **Balancing:** Addressed class imbalance (specifically rare "Cyborg" accounts) using Data Synthesis techniques.

* **Distribution:** 10,000 training samples and 2,500 test samples per category.

## 📈 Performance Results

| Model | Task | Test Accuracy | Test Loss |
|-------|------|---------------|-----------|
| **ResNet50** | Face Real/Fake | 92.33% | 0.1742 |
| **VGG16** | Face Real/Fake | 58.26% | 1.778 |
| **x-bot-profile** | Profile Type | 95.59% | N/A |

## 🚀 Deployment & Workflow

The system is deployed as a Streamlit web application.

### User Workflow

1. **Upload:** User uploads a face image and a Twitter profile screenshot.

2. **Analysis:** Images are preprocessed and sent to the AI models.

3. **Prediction:**
   * **Face:** Probability of Real vs. Fake.
   * **Profile:** Probability of Verified, Real, Cyborg, or Bot.

4. **Visualization:** Results displayed via risk heat meters and probability pie charts.

### Interpreting Results

* **Real Profile + AI Face:** Legitimate user likely using an avatar for privacy or fun.

* **Cyborg/Bot Profile + Real Face:** Suspicious account (potential stolen identity, fan page, or semi-automated).

* **Cyborg/Bot Profile + AI Face:** High risk. Likely part of a malicious bot network or spam campaign.

## 🔮 Future Roadmap

* **Multi-Platform Support:** Expand analysis to Instagram, Facebook, and LinkedIn by retraining models on platform-specific UI structures and metadata.

* **Frequency Domain Analysis:** Implement Discrete Cosine Transform (DCT) or FFT to detect GAN-specific noise invisible in pixel space.

* **Feature Fusion:** Combine spatial features (EfficientNetV2) with frequency features for richer representation.

## 👥 Team

**Supervision:** Eng. Aya Nada

**Members:**
* Faris Ahmed
* Ahmed Kamal
* Yara Harpy
