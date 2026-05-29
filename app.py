import streamlit as st
import torch
from PIL import Image

from models import (
    build_cnn_model,
    ResNet18WithDCT,
    predict_with_cnn,
    predict_with_cnn_dct,
)

st.set_page_config(
    page_title="Anime AI vs Human Detector",
    page_icon="🖼️",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_MODEL_PATH = "best_resnet18_anime_ai_detector.pth"
CNN_DCT_MODEL_PATH = "best_resnet18_dct_anime_ai_detector.pth"


@st.cache_resource
def load_cnn_model():
    model = build_cnn_model(num_classes=2)
    state_dict = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_cnn_dct_model():
    model = ResNet18WithDCT(num_classes=2, dct_input_dim=256)
    state_dict = torch.load(CNN_DCT_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def reset_prediction():
    st.session_state.prediction_result = None


st.markdown("""
<style>
    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    div[data-testid="stToolbar"] {
        right: 1rem;
    }

    .app-title {
        font-size: clamp(2rem, 3vw, 3rem);
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
        color: #f8fafc;
        word-break: break-word;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-bottom: 1.4rem;
    }

    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.1rem 1.1rem 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.9rem;
    }

    .result-box {
        border-radius: 14px;
        padding: 0.95rem 1rem;
        font-weight: 800;
        margin-bottom: 1rem;
        font-size: 1rem;
    }

    .result-ai {
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #ffd7d7;
    }

    .result-human {
        background: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.45);
        color: #d7ffe5;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 0.55rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .prob-value {
        font-size: 0.92rem;
        color: #e2e8f0;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    div[data-testid="stImage"] img {
        border-radius: 16px;
        width: 100%;
        object-fit: contain;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
        margin-top: 0.35rem;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    section[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.9rem 1rem;
        border-radius: 16px;
        margin: 0.4rem 0 0.9rem 0;
    }

    @media (max-width: 1100px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

st.markdown('<div class="app-title">Anime AI vs Human Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload gambar anime, pilih model, lalu lihat hasil prediksi</div>',
    unsafe_allow_html=True
)

left_col, mid_col, right_col = st.columns([0.95, 1.15, 0.9], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pengaturan</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Pilih model",
        ["CNN", "CNN + DCT"],
        key="model_choice",
        on_change=reset_prediction
    )

    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploaded_file",
        on_change=reset_prediction
    )

    predict_clicked = st.button("Prediksi", use_container_width=True)

    st.markdown(
        '<div class="small-note">Kalau model atau gambar diganti, hasil lama akan langsung direset.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

if predict_clicked:
    if image is None:
        st.session_state.prediction_result = None
        st.warning("Upload gambar dulu sebelum melakukan prediksi.")
    else:
        with st.spinner("Memproses prediksi..."):
            if model_choice == "CNN":
                model = load_cnn_model()
                result = predict_with_cnn(image, model, DEVICE)
            else:
                model = load_cnn_dct_model()
                result = predict_with_cnn_dct(image, model, DEVICE)

        st.session_state.prediction_result = {
            "model_choice": model_choice,
            "result": result
        }

with mid_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Preview Gambar</div>', unsafe_allow_html=True)

    if image is not None:
        st.image(image, use_container_width=True)
    else:
        st.info("Belum ada gambar yang diupload.")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    saved_result = st.session_state.prediction_result

    if saved_result is not None:
        result = saved_result["result"]
        active_model = saved_result["model_choice"]

        predicted_label = result["predicted_label"].lower()

        if predicted_label == "ai":
            st.markdown(
                '<div class="result-box result-ai">Prediksi: AI</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box result-human">Prediksi: HUMAN</div>',
                unsafe_allow_html=True
            )

        st.metric("Confidence", f'{result["confidence"] * 100:.2f}%')

        st.markdown('<div class="metric-label">Probabilitas AI</div>', unsafe_allow_html=True)
        st.progress(float(result["prob_ai"]))
        st.markdown(
            f'<div class="prob-value">{result["prob_ai"] * 100:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="metric-label">Probabilitas Human</div>', unsafe_allow_html=True)
        st.progress(float(result["prob_human"]))
        st.markdown(
            f'<div class="prob-value">{result["prob_human"] * 100:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="small-note">Model aktif: <b>{active_model}</b></div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Hasil prediksi akan muncul di sini setelah tombol Prediksi ditekan.")

    st.markdown('</div>', unsafe_allow_html=True)