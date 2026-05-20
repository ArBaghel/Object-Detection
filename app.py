import cv2
import numpy as np
import streamlit as st
import cvzone
import math
import pickle
import os
from ultralytics import YOLO

# --- PAGE UI CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Production Hub",
    page_icon="⚙️",
    layout="wide"
)

# --- CUSTOM CSS FOR STUNNING UI ---
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
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚙️ YOLOv8 Production Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time vision metrics displayed dynamically on-screen</div>", unsafe_allow_html=True)
st.divider() 

# --- LOADING THE SAVED FILES ---
PICKLE_FILE = "model_config.pkl"
WEIGHTS_FILE = "app_model_weights.pt"

@st.cache_resource
def load_saved_pipeline():
    if not os.path.exists(PICKLE_FILE) or not os.path.exists(WEIGHTS_FILE):
        return None, None
    with open(PICKLE_FILE, "rb") as pf:
        metadata = pickle.load(pf)
    model = YOLO(WEIGHTS_FILE)
    return model, metadata

model, config_data = load_saved_pipeline()

if model is None or config_data is None:
    st.error(f"⚠️ Critical Files Missing! Could not find '{WEIGHTS_FILE}' or '{PICKLE_FILE}' in this directory.")
    st.info("💡 **How to fix:** Run your `yolo-webcam.py` camera test screen by pressing **'q'** to let it successfully save the files.")
    st.stop()

# --- SIDEBAR INTERACTIVE SETTINGS ---
st.sidebar.header("🕹️ Config Settings")
st.sidebar.info(f"Loaded Profile: **{config_data['model_version']}**")

conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
run_webcam = st.sidebar.checkbox("🎥 Activate Live UI Stream", value=False)

class_names = config_data["class_names"]
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Classes")
with st.sidebar.expander("See Pickled Target Classes"):
    st.write(", ".join(list(class_names.values())))


# --- LIVE VIDEO STREAMING & DETECTION METRICS LOGIC ---
if run_webcam:
    st.subheader("🎬 Live Application Thread")
    
    # Create two interactive placeholders on our web canvas
    frame_placeholder = st.empty()
    
    st.markdown("### 📋 Live Detections Log")
    log_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, config_data["default_width"]) 
    cap.set(4, config_data["default_height"])  

    while run_webcam:
        success, img = cap.read()
        if not success:
            st.error("Unable to link or capture video source.")
            break

        # Flip image first for correct alignment
        img = cv2.flip(img, 1)
        results = model(img, stream=True, conf=conf_threshold)
        
        # Array to collect text detections found in the current frame
        current_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                
                # Format confidence into standard percentages (e.g., 0.98 -> 98%)
                raw_conf = box.conf[0].item()
                conf_pct = int(raw_conf * 100)
                
                cls = int(box.cls[0])
                label_text = class_names[cls].capitalize()

                # Save metadata for UI text container updates
                current_detections.append(f"{label_text} {conf_pct}%")

                # Overlay box text onto video stream canvas
                cvzone.putTextRect(
                    img, 
                    f'{label_text} {math.ceil(raw_conf * 100) / 100}',
                    (max(0, x1), max(35, y1)),
                    scale=0.8, thickness=1, offset=4
                )

        # 1. Update the main camera stream frame element
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)
        
        # 2. Update the dynamic information log right beneath the player
        if current_detections:
            # Build inline HTML badges for each object detected
            badge_html = "".join([f"<span class='detection-badge'>🔍 {d}</span>" for d in current_detections])
            log_placeholder.markdown(badge_html, unsafe_allow_html=True)
        else:
            log_placeholder.write("_No targets currently in frame above threshold._")

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info(" To start processing, expand the left sidebar and toggle on **'Activate Live UI Stream'**.")