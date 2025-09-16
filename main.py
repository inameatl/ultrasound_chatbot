import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import numpy as np
import streamlit as st
from torchcam.methods import SmoothGradCAMpp
from matplotlib import cm
import io
import os
from dotenv import load_dotenv
from groq import Groq

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------
# 1. Define the multitask model
# ---------------------------
class YOLOMultitask(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = resnet34(weights="IMAGENET1K_V1")
        layers = list(base.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # Segmentation head
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
# 2. Load model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOMultitask(num_classes=3).to(device)
model.load_state_dict(torch.load(
    r"yolo_multitask_busi1.pth",
    map_location=device
))
model.eval()
CLASS_NAMES = ["normal", "malignant", "benign"]

# Grad-CAM extractor
cam_extractor = SmoothGradCAMpp(model.encoder)

# ---------------------------
# 3. Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide")
st.title("🩺 Innovative Breast Ultrasound Analysis")
st.caption("Using **Grad-CAM + Segmentation ROI Fusion** for interpretable diagnosis")

with st.sidebar:
    st.header("⚙️ Settings")
    show_seg = st.checkbox("Show Segmentation Overlay", True)
    show_cam = st.checkbox("Show Grad-CAM Overlay", True)
    show_roi = st.checkbox("Show ROI Fusion", True)
    stacked_view = st.checkbox("Stacked View (instead of side by side)", False)

    st.markdown("---")
    st.subheader("📊 Dataset Info")
    st.info("""
    **Dataset:** BUSI (Breast Ultrasound Images)  
    - Classes: Normal, Benign, Malignant  
    - Includes segmentation masks  
    """)

def blend_overlay(base_img, overlay_img, alpha=0.5):
    """Blend overlay_img on top of base_img with transparency alpha."""
    return (alpha * overlay_img + (1 - alpha) * base_img).astype(np.uint8)

uploaded_file = st.file_uploader("📂 Upload an Ultrasound Image", type=["png", "jpg", "jpeg"])

pred_class, probs = None, None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    W, H = image.size

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Forward pass
    model.zero_grad()
    cls_out, seg_out, feat = model(img_tensor)

    # Classification probabilities
    probs = F.softmax(cls_out, dim=1).detach().cpu().numpy()[0]
    pred_class_idx = int(torch.argmax(cls_out, dim=1).item())
    pred_class = CLASS_NAMES[pred_class_idx]

    # ---------------------------
    # 1️⃣ If predicted class is normal → show original 50% size only
    # ---------------------------
    if pred_class == "normal":
        st.subheader("✅ Predicted Class: **Normal**")
        image_display = image.resize((W // 2, H // 2))
        st.image(image_display, caption="🖼️ Normal Ultrasound Image")
    else:
        # ---------------------------
        # 2️⃣ Non-normal → show probabilities
        # ---------------------------
        st.subheader(f"🔍 Predicted Class: **{pred_class}**")
        for i, cls_name in enumerate(CLASS_NAMES):
            st.progress(float(probs[i]), text=f"{cls_name}: {probs[i]*100:.2f}%")

        # ---------------------------
        # 3️⃣ Grad-CAM + Segmentation + ROI Fusion
        # ---------------------------
        cams_all = []
        for i in range(len(CLASS_NAMES)):
            model.zero_grad()
            cls_score = F.softmax(cls_out, dim=1)[0, i]
            cls_score.backward(retain_graph=True)
            cam = cam_extractor(i, cls_out)[0].cpu().detach().numpy().squeeze()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams_all.append(cam)

        cams_fused = np.zeros_like(cams_all[0])
        for i, cam in enumerate(cams_all):
            cams_fused += cam * probs[i]

        seg_mask = seg_out.squeeze().cpu().detach().numpy()
        seg_mask_bin = (seg_mask > 0.5).astype(np.float32)

        # Resize CAM to match segmentation
        cams_resized = Image.fromarray((cams_fused * 255).astype(np.uint8)).resize(seg_mask_bin.shape[::-1])
        cams_resized = np.array(cams_resized).astype(np.float32)

        # ROI Fusion
        roi_fused = cams_resized * seg_mask_bin
        roi_fused = (roi_fused - roi_fused.min()) / (roi_fused.max() - roi_fused.min() + 1e-8)

        # Resize overlays to original image size
        roi_color = cm.get_cmap('jet')(roi_fused)[:, :, :3]
        roi_color_img = Image.fromarray((roi_color * 255).astype(np.uint8)).resize((W, H))
        overlay_roi_full = blend_overlay(np.array(image), np.array(roi_color_img), alpha=0.5)
        overlay_roi = Image.fromarray(overlay_roi_full).resize((W // 2, H // 2))

        seg_resized = Image.fromarray((seg_mask_bin * 255).astype(np.uint8)).resize((W, H))
        overlay_seg_full = blend_overlay(np.array(image), np.stack([np.array(seg_resized)]*3, axis=-1), alpha=0.5)
        overlay_seg = Image.fromarray(overlay_seg_full).resize((W // 2, H // 2))

        cam_color = cm.get_cmap('jet')(cams_resized)[:, :, :3]
        cam_img = Image.fromarray((cam_color * 255).astype(np.uint8)).resize((W, H))
        overlay_cam_full = blend_overlay(np.array(image), np.array(cam_img), alpha=0.5)
        overlay_cam = Image.fromarray(overlay_cam_full).resize((W // 2, H // 2))

        # Display overlays
        st.subheader("🖼️ Visualization Overlays")
        if stacked_view:
            if show_roi: st.image(overlay_roi, caption="🔴 ROI Fused Overlay ")
            if show_seg: st.image(overlay_seg, caption="🟢 Segmentation Overlay ")
            if show_cam: st.image(overlay_cam, caption="🔵 Grad-CAM Overlay ")
        else:
            col1, col2, col3 = st.columns(3)
            if show_roi: col1.image(overlay_roi, caption="🔴 ROI Fused Overlay ")
            if show_seg: col2.image(overlay_seg, caption="🟢 Segmentation Overlay ")
            if show_cam: col3.image(overlay_cam, caption="🔵 Grad-CAM Overlay ")

        # Download ROI
        st.subheader("⬇️ Save Results (Full Resolution)")
        buf = io.BytesIO()
        Image.fromarray(overlay_roi_full).save(buf, format="PNG")
        st.download_button("Download ROI Image", data=buf.getvalue(),
                           file_name="roi_fused_overlay.png", mime="image/png")

        st.success("✅ Analysis Complete")

# ---------------------------
# 5. Chatbot Section (Groq LLM)
# ---------------------------
st.markdown("---")
st.header("🤖 AI Chatbot Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about the analysis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add prediction context
    context_msg = ""
    if pred_class is not None:
        prob_text = "\n".join([f"- {cls}: {p*100:.2f}%" for cls, p in zip(CLASS_NAMES, probs)])
        context_msg = f"Model prediction: {pred_class}\nClass probabilities:\n{prob_text}"

    # Query Groq LLM with supported model
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ supported model
        messages=[
            {"role": "system", "content": "You are a helpful medical AI assistant that explains breast ultrasound predictions in simple terms."},
            {"role": "system", "content": context_msg},
            *st.session_state.messages
        ],
        temperature=0.4,
        max_tokens=300
    )

    reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
