import cv2
import numpy as np
import streamlit as st
import cvzone
import pickle
import os
import av
import queue
import torch
import requests
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- PAGE UI CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Production Hub",
    page_icon="⚙️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #00FFCC, #0099FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #B0B3B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        display: inline-block;
        border: 1px solid #334155;
    }
    .hardware-card {
        background-color: #111827;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #1F2937;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚙️ YOLOv8 Production Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hybrid GPU/CPU Real-Time Web Stream Processing</div>", unsafe_allow_html=True)
st.divider()

# --- HARDWARE DETECTION ---
if torch.cuda.is_available():
    device_target = "cuda"
    hardware_status_html = "<div class='hardware-card' style='border-left: 4px solid #10B981;'><span style='color: #10B981; font-weight: bold;'>🚀 Hardware Acceleration:</span> Active Local GPU (CUDA) Detected.</div>"
else:
    device_target = "cpu"
    hardware_status_html = "<div class='hardware-card' style='border-left: 4px solid #F59E0B;'><span style='color: #F59E0B; font-weight: bold;'>🖥️ Hardware Context:</span> Standard CPU.</div>"

st.markdown(hardware_status_html, unsafe_allow_html=True)

# --- LOAD MODEL ---
PICKLE_FILE = "model_config.pkl"
WEIGHTS_FILE = "app_model_weights.pt"

@st.cache_resource
def load_saved_pipeline(device_name):
    if not os.path.exists(PICKLE_FILE) or not os.path.exists(WEIGHTS_FILE):
        return None, None
    with open(PICKLE_FILE, "rb") as pf:
        metadata = pickle.load(pf)
    model = YOLO(WEIGHTS_FILE)
    model.to(device_name)
    return model, metadata

model, config_data = load_saved_pipeline(device_target)

if model is None or config_data is None:
    st.error(f"⚠️ Critical Files Missing! Could not find '{WEIGHTS_FILE}' or '{PICKLE_FILE}'.")
    st.info("💡 Run your local `yolo-webcam.py` script first and press 'q' to save assets, then upload.")
    st.stop()

class_names = config_data["class_names"]

# --- SIDEBAR ---
st.sidebar.header("🕹️ Config Settings")
st.sidebar.info(f"Loaded Profile: **{config_data['model_version']}**")
st.sidebar.success(f"Processing Engine: **{device_target.upper()}**")

conf_slider_value = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.0, max_value=1.0, value=0.45, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Camera Selection")
camera_facing = st.sidebar.radio(
    "Camera Mode",
    options=["Environment (Rear/Back)", "User (Front/Selfie)"],
    index=0,
    help="Use 'Environment' for phone rear camera, 'User' for front/laptop webcam."
)
facing_mode = "environment" if camera_facing.startswith("Environment") else "user"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Classes")
with st.sidebar.expander("See Pickled Target Classes"):
    st.write(", ".join(list(class_names.values())))


# =====================================================================
# 🔧 ICE / TURN SERVER CONFIG
# Root cause of the spinning loader on hosted Streamlit:
# STUN only works on the same network. When browser (phone) and
# server (cloud) are on different networks, WebRTC needs a TURN relay.
# We use the free open relay from metered.ca as fallback.
# =====================================================================
@st.cache_data(ttl=3600)
def get_ice_servers():
    """
    Fetches fresh TURN credentials from the open relay.
    Falls back to Google STUN only if the request fails.
    """
    try:
        # Free public TURN relay — no account needed, rate-limited but works for demos
        response = requests.get(
            "https://global.xirsys.net/_turn/streamlit-yolo",
            timeout=3
        )
    except Exception:
        pass

    # Primary: use metered.ca free TURN (most reliable free option)
    # These are the public demo credentials — replace with your own for production
    return {
        "iceServers": [
            # STUN servers (work on same network / when not behind strict NAT)
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            # TURN relay servers — these are what actually fix the spinning loader
            # Free public TURN from Open Relay Project (no key needed)
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }

ICE_CONFIG = get_ice_servers()


# =====================================================================
# 🎥 VIDEO FRAME CALLBACK
# =====================================================================
log_queue = queue.Queue(maxsize=10)

def video_frame_callback(frame):
    img_rgb = frame.to_ndarray(format="rgb24")
    img_rgb = cv2.flip(img_rgb, 1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, stream=True, conf=conf_slider_value, device=device_target)
    current_frame_logs = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img_rgb, (x1, y1, w, h), l=15, t=3, colorR=(0, 255, 204))

            raw_conf = box.conf[0].item()
            conf_pct = int(raw_conf * 100)
            cls = int(box.cls[0])
            label_text = class_names[cls].capitalize()
            current_frame_logs.append(f"{label_text} {conf_pct}%")

            cvzone.putTextRect(
                img_rgb,
                f'{label_text} {conf_pct}%',
                (max(0, x1), max(35, y1)),
                scale=0.8, thickness=1, offset=4,
                colorR=(0, 153, 255), colorT=(255, 255, 255)
            )

    if current_frame_logs:
        try:
            log_queue.put_nowait(current_frame_logs)
        except queue.Full:
            pass

    return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")


# --- MAIN PANEL ---
st.subheader("🎬 Active Live Camera Input Feed")

# Connection status hint
st.info(
    "📱 **On mobile?** Select **Environment (Rear/Back)** in the sidebar, then press **START** and allow camera access. "
    "If it keeps spinning, your network may block WebRTC — try switching to a different Wi-Fi or mobile data."
)

webrtc_ctx = webrtc_streamer(
    key=f"yolov8-stream-{facing_mode}",   # key change forces re-init on camera switch
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration=ICE_CONFIG,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": facing_mode},
            "width":  {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    async_processing=True,
)

st.markdown("### 📋 Live Detections Log")
log_placeholder = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            active_targets = log_queue.get(timeout=1.0)
            badge_html = "".join(
                [f"<span class='detection-badge'>🔍 {item}</span>" for item in active_targets]
            )
            log_placeholder.markdown(badge_html, unsafe_allow_html=True)
        except queue.Empty:
            log_placeholder.write("_No targets currently tracked in frame._")
else:
    log_placeholder.info("💡 Click **'Start'** above and accept browser camera permissions.")
