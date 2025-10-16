import os
import sys
import torch
import streamlit as st
from PIL import Image 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from config import DEVICE, OUTPUTS_DIR
from models.model_loader import load_model, infer
from explainers.captum_explainers import generate_explanation
from utils.viz import overlay_heatmap
from metrics.xai_metrics import faithfulness_deletion, sensitivity
from utils.logger import get_logger

logger = get_logger("dashboard")
st.set_page_config(page_title="Aletheia XAI", layout="wide")

st.title("🧠 aletheia; explainable AI dashboard")
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #a8d5ff !important;
        }
        .metric-card {
            background: #1e1e26;
            padding: 1.2em;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### upload a visual image to see how aletheia explains your model’s reasoning")

st.sidebar.header("settings")
explainer_choice = st.sidebar.selectbox(
    "select Explainer", ["gradcam", "integrated_gradients", "deeplift"], index=0
)
model_choice = st.sidebar.selectbox("select model", ["resnet50"], index=0)

os.makedirs(OUTPUTS_DIR, exist_ok=True)

@st.cache_resource(show_spinner=True)
def get_model(model_name):
    with st.spinner(f"Loading {model_name}..."):
        model, preprocess, _ = load_model(model_name)
    return model, preprocess

model, preprocess = get_model(model_choice)

uploaded_file = st.file_uploader("upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="uploaded Image", use_column_width=True)

    with st.spinner("running model inference..."):
        predicted_class, confidence = infer(model, preprocess, image)
    st.success("inference complete!")

    st.markdown(f"### prediction: class **{predicted_class}** (confidence: {confidence:.4f})") 

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE) # type: ignore

    with st.spinner(f"generating explanation using {explainer_choice}..."):
        try:
            attr_map = generate_explanation(model, input_tensor, explainer=explainer_choice, target=predicted_class) # type: ignore
        except Exception as e:
            st.error(f"error with {explainer_choice}: {e}")
            st.stop()

    overlay_path = os.path.join(OUTPUTS_DIR, f"overlay_{uploaded_file.name}")
    overlayed_image = overlay_heatmap(image, attr_map, save_path=overlay_path)

    st.markdown("## attribution heatmap")
    st.image(overlayed_image, caption=f"Explainer: {explainer_choice}", use_column_width=True)

    with st.spinner("computing XAI metrics..."):
        perturb = input_tensor + 0.01 * torch.randn_like(input_tensor)
        attr_map_perturbed = generate_explanation(model, perturb, explainer=explainer_choice, target=predicted_class) # type: ignore
        faith_score = faithfulness_deletion(model, input_tensor, attr_map, target=predicted_class)
        sens_score = sensitivity(attr_map, attr_map_perturbed)

    st.markdown("## XAI metrics")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"<div class='metric-card'><h3>Faithfulness</h3><p>{faith_score:.4f}</p></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-card'><h3>Sensitivity</h3><p>{sens_score:.6f}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**done!** your results are saved under the `/outputs` directory")
else:
    st.info("upload an image to begin analysis")
