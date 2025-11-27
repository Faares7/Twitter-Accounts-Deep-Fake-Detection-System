import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, SiglipForImageClassification

# ==============================
# DEVICE SETUP
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# FACE MODEL SETUP
# ==============================
class ResNetClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2):
        super().__init__()
        from timm import create_model
        self.backbone = create_model(model_name, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# Load face model
face_model = ResNetClassifier(model_name='resnet50')
checkpoint = torch.load(r"models\final_deepfake_detector.pth", map_location=device)
face_model.load_state_dict(checkpoint['model_state_dict'])
face_model.to(device)
face_model.eval()

face_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_face(image: Image.Image):
    img_t = face_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = face_model(img_t)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    label = "REAL" if pred_idx == 0 else "FAKE"

    return {
        "label": label,
        "real_prob": float(probs[0]),
        "fake_prob": float(probs[1])
    }

# ==============================
# SCREENSHOT MODEL SETUP
# ==============================
screenshot_model_name = "prithivMLmods/x-bot-profile-detection"
screenshot_model = SiglipForImageClassification.from_pretrained(screenshot_model_name)
screenshot_processor = AutoImageProcessor.from_pretrained(screenshot_model_name)
screenshot_model.eval()

id2label = {
    0: "bot",
    1: "cyborg",
    2: "real",
    3: "verified"
}

def predict_screenshot(image: Image.Image):
    inputs = screenshot_processor(
        images=[image],
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = screenshot_model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0).tolist()

    return {id2label[i]: round(probs[i], 3) for i in range(len(probs))}

# ==============================
# STREAMLIT UI (Upgraded Version)
# ==============================

st.set_page_config(page_title="X (Twitter) Profile Analyzer", layout="wide")

# ----- Custom CSS -----
st.markdown("""
<style>
body {
    background-color: #0d1117;
    color: #c9d1d9;
}
h1, h2, h3, h4 {
    color: #58a6ff;
}
.card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: bold;
}
.face-real {background-color: #28a745; color:white;}
.face-fake {background-color: #dc3545; color:white;}
.ss-verified {background-color: #28a745; color:white;}
.ss-real {background-color: #1f77b4; color:white;}
.ss-cyborg {background-color: #f1c40f; color:black;}
.ss-bot {background-color: #dc3545; color:white;}
.prob-bar {
    height: 20px;
    border-radius: 5px;
    margin-bottom: 5px;
}
.heat-bar {
    height: 20px;
    border-radius: 10px;
    background: linear-gradient(to right, #28a745 0%, #f1c40f 50%, #dc3545 100%);
    position: relative;
}
.heat-bar-ss {
    height: 20px;
    border-radius: 10px;
    background: linear-gradient(to right, #1f77b4 0%, #28a745 35%, #f1c40f 70%, #dc3545 100%);
    position: relative;
}
.cursor {
    position: absolute;
    top: -5px;
    width: 2px;
    height: 30px;
    background-color: white;
}
.explain-box {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #c9d1d9;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Page title
st.title("🖼️ X (Twitter) Profile Analyzer")

st.markdown("""
### 🔍 About This Project
This tool analyzes **Twitter profile screenshots** and **face images** using two specialized AI models.  
The goal is to detect whether a social media account is **authentic, AI‑generated, stolen identity, or bot‑like** by combining:
- Face deepfake detection  
- Profile classification (real / verified / cyborg / bot)  
- Behavioral and visual explainability  

The system produces a final interpretation that helps you understand  
**how trustworthy the account appears**.
""")
st.markdown("Upload a face image and a Twitter profile screenshot to analyze them.")

# Upload images
col1, col2 = st.columns(2)
with col1:
    face_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
with col2:
    ss_file = st.file_uploader("Upload Twitter Screenshot", type=["jpg", "jpeg", "png"])

def generate_final_insight(face_label, profile_label):
    face_real = (face_label == "REAL")
    profile_realish = (profile_label in ["real", "verified"])
    profile_fakeish = (profile_label in ["cyborg", "bot"])

    # CASE 1 — Profile real/verified + face real
    if profile_realish and face_real:
        return """
### 🟢 Most Likely a Real Account

**Why?**
- The profile appears **legitimate** (Real or Verified)
- The face shows **no deepfake or AI‑generated artifacts**
- Both signals align strongly toward authenticity

**Interpretation:**  
This combination strongly suggests this is a **genuine, human‑run account** with very low risk.
"""

    # CASE 2 — Profile real/verified + face fake
    if profile_realish and not face_real:
        return """
### 🟡 Real Profile, AI‑Generated Face

**Why?**
- The profile looks legitimate, but the **face appears AI‑generated**
- Deepfake artifacts or face‑synthesis patterns were detected
- Behavior does NOT match bot activity

**Possible Explanations:**
- A user using an **AI avatar** for fun  
- Someone hiding their identity  
- A privacy-conscious user using generated faces

**Interpretation:**  
Not malicious, but the face is **not real**.
"""

    # CASE 3 — Profile bot/cyborg + face real
    if profile_fakeish and face_real:
        return """
### 🟡 Suspicious: Real Face + Fake‑Like Profile

**Why?**
- The profile shows **bot‑like or cyborg behavior**
- The face appears to be a real human photo
- This mismatch is common in deceptive accounts

**Possible Scenarios:**
- A **stolen identity**  
- A **fan page** using someone else’s real picture  
- A **semi‑automated human‑assisted account**

**Interpretation:**  
Treat this with **caution** — the face is real, but the account behavior is not.
"""

    # CASE 4 — Profile bot/cyborg + face fake
    if profile_fakeish and not face_real:
        return """
### 🔴 High Risk: Fully Fake Account

**Why?**
- The profile behaves like a **bot/cyborg**
- The face is **AI‑generated / deepfake**
- No component of the identity appears authentic

**Strong Indicators Of:**
- A completely fabricated identity  
- A bot network account  
- Possible phishing, spam, or malicious automation  

**Interpretation:**  
This is a **high‑risk fake account**. Avoid interacting with it.
"""

    return "Unable to generate insight."




if face_file and ss_file:
    # Open and resize images
    face_img = Image.open(face_file).convert("RGB").resize((250,250))
    ss_img = Image.open(ss_file).convert("RGB").resize((250,250))

    # Predict
    face_result = predict_face(face_img)
    screenshot_result = predict_screenshot(ss_img)

    # Columns for images + pie charts
    col3, col4 = st.columns(2)

    # --- FACE COLUMN ---
    with col3:
        st.subheader("Face Detection")
        st.image(face_img, caption="Face Image", width=250)

        # Pie chart for face
        face_labels = ["REAL", "FAKE"]
        face_probs = [face_result['real_prob'], face_result['fake_prob']]
        dominant_face = face_labels[face_probs.index(max(face_probs))]
        face_colors = ["#28a745", "#dc3545"]

        fig_face = go.Figure(data=[go.Pie(
            labels=face_labels,
            values=face_probs,
            marker_colors=face_colors,
            textinfo='label+percent',
            hole=0.3
        )])
        fig_face.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_face, use_container_width=True)

        # Risk Heat Meter
        risk_score = face_probs[1]  # FAKE probability
        st.markdown("**Risk Heat Meter (Deepfake Probability)**")
        st.markdown(f"""
        <div class='heat-bar'>
            <div class='cursor' style='left:{risk_score*100}%;'></div>
        </div>
        """, unsafe_allow_html=True)

        # Explainability Section (dynamic)
        st.subheader("Explainability")
        if dominant_face == "REAL":
            explain_texts = [
                "Face appears natural with typical skin texture.",
                "Eyes and facial geometry match real patterns.",
                "No detectable GAN or deepfake artifacts."
            ]
        else:
            explain_texts = [
                "Face has unusually smooth skin → possible deepfake.",
                "Eyes/facial geometry not typical of natural images.",
                "High similarity to common GAN artifacts."
            ]
        for text in explain_texts:
            st.markdown(f"<div class='explain-box'>{text}</div>", unsafe_allow_html=True)

    # --- SCREENSHOT COLUMN ---
    with col4:
        st.subheader("Twitter Profile Detection")
        st.image(ss_img, caption="Twitter Screenshot", width=250)

        ss_labels = list(screenshot_result.keys())
        ss_probs = list(screenshot_result.values())
        dominant_ss = ss_labels[ss_probs.index(max(ss_probs))]

        ss_colors = {
            "verified": "#28a745",
            "real": "#1f77b4",
            "cyborg": "#f1c40f",
            "bot": "#dc3545"
        }

        fig_ss = go.Figure(data=[go.Pie(
            labels=ss_labels,
            values=ss_probs,
            marker_colors=[ss_colors[label] for label in ss_labels],
            textinfo='label+percent',
            hole=0.3
        )])
        fig_ss.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_ss, use_container_width=True)

        # Screenshot Risk Heat Meter
        risk_score_ss = (
            0.0 * screenshot_result.get("real",0) +
            0.33 * screenshot_result.get("verified",0) +
            0.66 * screenshot_result.get("cyborg",0) +
            1.0 * screenshot_result.get("bot",0)
        )
        st.markdown("**Account Risk Meter**")
        st.markdown(f"""
        <div class='heat-bar-ss'>
            <div class='cursor' style='left:{risk_score_ss*100}%;'></div>
        </div>
        """, unsafe_allow_html=True)


        # Explainability Section (dynamic)
        st.subheader("Explainability")
        if dominant_ss == "verified":
            explain_texts_ss = [
                "Profile verified → low risk.",
                "Bio completeness is sufficient.",
                "Posting pattern typical of human accounts."
            ]
        elif dominant_ss == "real":
            explain_texts_ss = [
                "Profile appears real but not verified.",
                "Normal bio and posting patterns.",
                "No strong bot-like signals."
            ]
        elif dominant_ss == "cyborg":
            explain_texts_ss = [
                "Account shows mixed human/bot behavior.",
                "Posting frequency higher than normal.",
                "Profile picture may be AI-generated."
            ]
        else:  # bot
            explain_texts_ss = [
                "High likelihood of bot activity.",
                "Incomplete bio and unusual posting frequency.",
                "Profile picture likely AI-generated or repeated."
            ]
        for text in explain_texts_ss:
            st.markdown(f"<div class='explain-box'>{text}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📌 Final Interpretation")

    face_label = dominant_face
    profile_label = dominant_ss

    final_text = generate_final_insight(face_label, profile_label)
    st.markdown(final_text, unsafe_allow_html=True)