import streamlit as st
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34

# ---------------------------
# Load Groq API key
# ---------------------------
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------
# Model Definition
# ---------------------------
class YOLOMultitask(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = resnet34(weights="IMAGENET1K_V1")
        layers = list(base.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        cls_out = self.fc(self.avgpool(feat).view(feat.size(0), -1))
        seg_out = self.seg_head(feat)
        return cls_out, seg_out, feat

# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOMultitask(num_classes=3).to(device)
model.load_state_dict(torch.load(
    r"E:\Ultrsound project\multitask\yolo_multitask_busi1.pth",
    map_location=device
))
model.eval()
CLASS_NAMES = ["normal", "malignant", "benign"]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🩺 AI Breast Ultrasound Tutor", layout="wide")
st.markdown("<h1 style='text-align:center; color:#D32F2F;'>🩺 AI Tutor: Breast Ultrasound Self-Learning</h1>", unsafe_allow_html=True)
st.markdown("---")

# Instructions in an expander
with st.expander("📖 Instructions"):
    st.markdown("""
    1. Upload a breast ultrasound image (normal, benign, or malignant).  
    2. Optionally segment the lesion using sliders.  
    3. Predict the lesion class (benign/malignant).  
    4. Chat with AI Tutor for guidance.  
    5. Feedback includes textual guidance and Dice score for segmentation.  
    """)

# ---------------------------
# Dice coefficient
# ---------------------------
def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

# ---------------------------
# System prompt for AI Tutor
# ---------------------------
system_content = """
You are an AI medical tutor specializing in breast ultrasound images. 
Assist students in a self-learning workflow:
- Upload breast ultrasound images (normal, benign, malignant)
- Optionally draw segmentation masks
- Predict lesion class

Guidelines:
1. Focus on breast ultrasound images.
2. Never provide exact answers automatically; give guidance, hints, motivation.
3. Provide feedback on Dice score and class predictions.
4. Encourage self-learning.
"""

# ---------------------------
# Image Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png","jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((128,128))
    st.image(image, caption="Uploaded Image (128x128)", use_column_width=False)

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # ---------------------------
    # Model Prediction
    # ---------------------------
    model.zero_grad()
    cls_out, seg_out, feat = model(img_tensor)

    seg_mask = seg_out.squeeze().cpu().detach().numpy()
    seg_mask_resized = np.array(Image.fromarray((seg_mask * 255).astype(np.uint8)).resize((128,128)))
    ground_truth_mask = (seg_mask_resized > 127).astype(np.uint8)

    pred_class_idx = int(torch.argmax(cls_out, dim=1).item())
    ground_truth_class = CLASS_NAMES[pred_class_idx]

    probs = F.softmax(cls_out, dim=1).detach().cpu().numpy()[0]

    # ---------------------------
    # Segmentation by Student
    # ---------------------------
    do_segmentation = st.checkbox("🖌 I want to segment the lesion")
    student_mask = np.zeros((128,128), dtype=np.uint8)
    if do_segmentation:
        st.subheader("Draw your segmentation manually")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.slider("X1 (Left)", 0, 127, 20)
                x2 = st.slider("X2 (Right)", 0, 127, 100)
            with col2:
                y1 = st.slider("Y1 (Top)", 0, 127, 20)
                y2 = st.slider("Y2 (Bottom)", 0, 127, 100)

        student_mask[y1:y2, x1:x2] = 1
        overlay_img = np.array(image).copy()
        overlay_img[student_mask==1] = [255, 255, 0]  # yellow overlay
        st.image(overlay_img, caption="Your segmentation (yellow)", use_column_width=False)

    # ---------------------------
    # Chat Interface
    # ---------------------------
    st.subheader("💬 Chat with AI Tutor")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask AI Tutor questions or predict class..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        seg_feedback = ""
        if do_segmentation:
            dice = dice_coef(ground_truth_mask, student_mask)
            if dice > 0.8:
                seg_feedback = f"🎉 Excellent! Your segmentation Dice score is {dice:.2f}"
            elif dice > 0.5:
                seg_feedback = f"👍 Partial match. Dice score: {dice:.2f}"
            else:
                seg_feedback = f"⚠️ Segmentation is off. Dice score: {dice:.2f}"

        class_feedback = ""
        pred_user_class = prompt.lower()
        if pred_user_class in ["benign", "malignant"]:
            if pred_user_class == ground_truth_class:
                class_feedback = f"🎉 Correct! Your class prediction ({pred_user_class}) is accurate."
            else:
                class_feedback = f"❌ Not quite. Model predicted {ground_truth_class}. Review features: shape, margins, posterior acoustic features, echogenicity."

        # Groq AI Tutor response
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":system_content},
                {"role":"system","content":seg_feedback},
                {"role":"system","content":class_feedback},
                *st.session_state.messages
            ],
            temperature=0.4,
            max_tokens=300
        )

        reply = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})

    st.success("✅ Self-learning session active. Chat and optionally segment for feedback.")
