"""
Fabric Defect Detection & Model Evaluation Dashboard.
Real-time camera inspection, model evaluation, and live defect alerts.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import json
import pandas as pd
import streamlit as st

from dashboard.runtime import get_runtime
from ml.video_inspector import ClassicalMLVideoInspector, run_comparative_video_study

# Page Configuration
st.set_page_config(
    page_title="Fabric Defect Inspector AI",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Stat Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alert Banners */
    .alert-banner-danger {
        background: linear-gradient(90deg, rgba(218, 54, 51, 0.9) 0%, rgba(248, 81, 73, 0.95) 100%);
        color: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        font-weight: 700;
        font-size: 19px;
        text-align: center;
        box-shadow: 0 0 25px rgba(248, 81, 73, 0.6);
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { transform: scale(1.0); }
        100% { transform: scale(1.015); }
    }
    
    .alert-banner-normal {
        background: rgba(35, 134, 54, 0.2);
        border: 1px solid rgba(46, 160, 67, 0.5);
        color: #3fb950;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 15px;
        text-align: center;
    }
    
    /* Video Box */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #30363d;
        background: #000;
    }
</style>
""", unsafe_allow_html=True)

# Audio alert generator
AUDIO_ALERT_HTML = """
<audio autoplay>
    <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU{data}" type="audio/wav">
</audio>
"""

# Access Runtime Singleton
runtime = get_runtime()

# ─────────────────── Sidebar Controls ────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Inspection Configuration")
    
    # Session Details
    fabric_id = st.text_input("Fabric / Roll ID", value="Roll-A-01")
    fabric_length = st.number_input("Fabric Length (Metres)", min_value=1.0, value=15.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 📹 Camera Feed")
    cam_option = st.selectbox(
        "Camera Source",
        options=[
            "Synthetic Simulation (Test Weave & Datasets)",
            "Webcam 0 (Default Laptop / USB Camera)",
            "Webcam 1 (Secondary Camera)",
            "Phone Stream / RTSP URL"
        ],
        index=0
    )
    
    source_val = "synthetic"
    if cam_option == "Webcam 0 (Default Laptop / USB Camera)":
        source_val = "0"
    elif cam_option == "Webcam 1 (Secondary Camera)":
        source_val = "1"
    elif cam_option == "Phone Stream / RTSP URL":
        source_val = st.text_input("Stream URL", value="http://192.168.1.10:8080/video")
        
    st.markdown("---")
    st.markdown("### 🧠 Model Selection")
    model_choice = st.selectbox(
        "Classifier Architecture",
        options=[
            "YOLO11n-cls (Recommended: 89.96% Acc, 1.58ms)",
            "YOLO11s-cls (Standard: 88.89% Acc, 2.17ms)",
            "YOLOv8s-cls (Prior Gen: 87.63% Acc, 1.54ms)",
            "Custom Weights Path"
        ],
        index=0
    )
    
    if "YOLO11n-cls" in model_choice:
        weights_path = "models/yolo11n_classifier_best.pt" if Path("models/yolo11n_classifier_best.pt").exists() else "runs/classify/runs/cls_yolo11n/weights/best.pt"
    elif "YOLO11s-cls" in model_choice:
        weights_path = "models/fabric_classifier_best.pt" if Path("models/fabric_classifier_best.pt").exists() else "runs/cls/weights/best.pt"
    elif "YOLOv8s-cls" in model_choice:
        weights_path = "models/cls_yolov8s_best.pt" if Path("models/cls_yolov8s_best.pt").exists() else "runs/classify/runs/cls_yolov8s/weights/best.pt"
    else:
        weights_path = st.text_input("Weights File", value="models/fabric_classifier_best.pt")
        
    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.95, value=0.50, step=0.05)
    
    lighting_corr = st.selectbox(
        "Lighting Normalization",
        options=["clahe", "illumination_norm", "combined", "none"],
        index=0,
        help="Applies contrast/illumination correction before model inference"
    )
    
    beep_alert = st.checkbox("System Audio Beep on Defect", value=True)
    
    st.markdown("---")
    # Action buttons
    snap = runtime.snapshot()
    is_running = snap["running"]
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ START", use_container_width=True, disabled=is_running, type="primary"):
            try:
                runtime.start_session(
                    fabric_id=fabric_id,
                    fabric_length_m=fabric_length,
                    source=source_val,
                    weights=weights_path,
                    mock_mode=(source_val == "synthetic"),
                    conf=conf_thresh,
                    lighting=lighting_corr,
                    beep=beep_alert
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start: {e}")
                
    with col_s2:
        if st.button("⏹ STOP", use_container_width=True, disabled=not is_running):
            runtime.stop_session()
            st.rerun()

# ─────────────────── Main Header ─────────────────────────────────────────────
snap = runtime.snapshot()
is_running = snap["running"]
stats = snap["stats"]
alert = snap["alert"]
counts = snap["class_counts"]
total_defects = sum(counts.values())

st.markdown(f"""
<div class="header-box">
    <div>
        <h2 style="margin: 0; font-size: 24px; color: #58a6ff;">🧵 Autonomous Fabric Defect Inspection System</h2>
        <span style="color: #8b949e; font-size: 14px;">Edge Computer Vision • Real-Time Detection • RTX 3050 GPU Acceleration</span>
    </div>
    <div style="text-align: right;">
        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: {'#2ea043' if is_running else '#8b949e'}; margin-right: 6px;"></span>
        <strong style="color: {'#3fb950' if is_running else '#8b949e'}; font-size: 15px;">{'SYSTEM INSPECTING' if is_running else 'SYSTEM STANDBY'}</strong>
        <div style="color: #8b949e; font-size: 12px;">Session: {snap.get('session_id') or 'None'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────── Main Tabs ───────────────────────────────────────────────
tab_video_ml, tab_live, tab_eval, tab_history, tab_sessions = st.tabs([
    "🎞️ Upload Video (Classical ML Study)",
    "📹 Live Camera & YOLO Pipeline",
    "📊 Model Evaluation & Benchmarks",
    "📸 Defect Snapshots & Event Log",
    "📁 Session Records & Export"
])

# ─────────────────── Tab 1: Upload Video (Classical ML Study) ────────────────
with tab_video_ml:
    st.markdown("""
    <div style="background: rgba(22, 27, 34, 0.7); border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #58a6ff;">🎞️ Offline Video Inspection & Classical ML Comparative Study</h3>
        <p style="margin: 6px 0 0 0; color: #8b949e; font-size: 14px;">
            Inspect recorded fabric videos (simulating line-scan camera / panning fabric). Uses <strong>handcrafted feature descriptors (GLCM texture, LBP, HOG edges, statistical moments)</strong> 
            and <strong>3 Classical Machine Learning models (Random Forest, SVM, Gradient Boosting)</strong> with automated multi-frame defect tracking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_v1, col_v2 = st.columns([1, 1], gap="large")

    with col_v1:
        st.markdown("#### 1. Select Video Input")
        video_source_type = st.radio(
            "Video Source:",
            options=["Built-in Moving Fabric Scan (Demo)", "Upload New Fabric Video (.mp4, .avi, .mov)"],
            horizontal=True
        )

        video_file_path = None
        if video_source_type == "Built-in Moving Fabric Scan (Demo)":
            demo_path = Path("data/fabric_moving_scan.mp4")
            if not demo_path.exists():
                from ml.create_demo_video import generate_panning_fabric_video
                generate_panning_fabric_video()
            video_file_path = str(demo_path)
            st.info(f"Loaded demo video: `{demo_path}` (Line-scan panning weave with hole, stain, and thread anomalies)")
        else:
            uploaded_video = st.file_uploader("Upload Fabric Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                temp_dir = Path("logs/uploaded_videos")
                temp_dir.mkdir(parents=True, exist_ok=True)
                video_file_path = str(temp_dir / uploaded_video.name)
                with open(video_file_path, "wb") as f:
                    f.write(uploaded_video.read())
                st.success(f"Video uploaded: `{uploaded_video.name}` ({uploaded_video.size / 1024 / 1024:.2f} MB)")

    with col_v2:
        st.markdown("#### 2. Model & Inspection Settings")
        ml_model_choice = st.selectbox(
            "Classical Machine Learning Model:",
            options=[
                "🏆 Comparative Study (Run All 3 Models Side-by-Side)",
                "Random Forest (Ensemble Trees)",
                "Support Vector Machine (SVM - RBF Kernel)",
                "Gradient Boosting (HistGBM)"
            ],
            index=0
        )
        conf_sens = st.slider("Detection Sensitivity (Confidence Threshold)", min_value=0.20, max_value=0.80, value=0.35, step=0.05,
                              help="Lower threshold detects subtle / faint anomalies; higher threshold requires strong defect contrast.")
        
        run_inspection_btn = st.button("🚀 Run Video Inspection", use_container_width=True, type="primary")

    if run_inspection_btn:
        if not video_file_path or not Path(video_file_path).exists():
            st.error("Please provide or upload a valid video file first.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)

            if "Comparative Study" in ml_model_choice:
                status_text.text("Running Comparative Study across Random Forest, SVM, and Gradient Boosting...")
                comp_res = run_comparative_video_study(video_file_path, conf_threshold=conf_sens, progress_cb=update_progress)
                progress_bar.progress(1.0)
                status_text.success("Comparative Study Finished Successfully!")

                st.markdown("---")
                st.markdown("### 🏆 Comparative Study Results (Across 3 Classical ML Models)")
                
                comp_df = comp_res["comparative_df"]
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Charts
                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    st.markdown("##### ⚡ Processing Throughput (FPS)")
                    st.bar_chart(comp_df.set_index("ML Model")["Processing Speed (FPS)"])
                with ch_col2:
                    st.markdown("##### 🔍 Total Defects Detected")
                    st.bar_chart(comp_df.set_index("ML Model")["Total Defects Found"])

                # Offline Benchmark Metrics
                metrics_json_path = Path("models/ml_comparison_metrics.json")
                if metrics_json_path.exists():
                    st.markdown("##### 📋 Offline Validation Metrics (Held-out Test Dataset)")
                    with open(metrics_json_path) as mf:
                        m_data = json.load(mf)
                    b_rows = []
                    for m_k, m_v in m_data.items():
                        b_rows.append({
                            "Model": m_k,
                            "Accuracy": f"{m_v['accuracy']*100:.2f}%",
                            "Macro F1": f"{m_v['macro_f1']:.4f}",
                            "Precision": f"{m_v['precision']:.4f}",
                            "Recall": f"{m_v['recall']:.4f}",
                            "Latency (ms/patch)": f"{m_v['inference_ms_per_patch']:.3f} ms",
                            "Model Size": f"{m_v['model_size_kb']:.1f} KB"
                        })
                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

                # Video playback with model selector
                st.markdown("#### 🎬 Annotated Inspection Video")
                view_col1, _ = st.columns([2, 2])
                with view_col1:
                    vid_model = st.selectbox(
                        "Choose Annotated Model Video to Play:",
                        ["Random Forest", "Support Vector Machine (SVM)", "Gradient Boosting (HistGBM)"]
                    )
                sel_model_res = comp_res["model_results"].get(vid_model, top_model_res)
                out_vid = sel_model_res.get("output_video_path")
                if out_vid and Path(out_vid).exists():
                    try:
                        with open(out_vid, "rb") as vf:
                            st.video(vf.read(), format="video/mp4")
                    except Exception:
                        st.video(out_vid)

                st.markdown("#### 📝 Tracked Defect Events Timeline")
                ev_df = sel_model_res["events_df"]
                if not ev_df.empty:
                    st.dataframe(ev_df.drop(columns=["Snapshot"], errors="ignore"), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No persistent defects flagged by {vid_model} in this video clip.")

            else:
                # Single Model Run
                cleaned_name = ml_model_choice.split(" (")[0]
                status_text.text(f"Running inspection with {cleaned_name}...")
                inspector = ClassicalMLVideoInspector(model_name=cleaned_name)
                out_vid = f"logs/annotated_single_{cleaned_name.replace(' ', '_').lower()}.mp4"
                res = inspector.inspect_video(
                    video_path=video_file_path,
                    output_video_path=out_vid,
                    conf_threshold=conf_sens,
                    progress_cb=update_progress
                )
                progress_bar.progress(1.0)
                status_text.success(f"{cleaned_name} Inspection Completed!")

                st.markdown("---")
                st.markdown(f"### 📊 Inspection Summary: {cleaned_name}")
                s = res["stats"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Defects", s["total_unique_defects"])
                m2.metric("Processing Speed", f"{s['processing_fps']:.1f} FPS")
                m3.metric("Avg Latency", f"{s['average_ms_per_frame']:.1f} ms")
                m4.metric("Video Duration", f"{s['video_duration_sec']} s")

                # Video display
                if out_vid and Path(out_vid).exists():
                    st.markdown("#### 🎬 Annotated Output Video")
                    try:
                        with open(out_vid, "rb") as vf:
                            st.video(vf.read(), format="video/mp4")
                    except Exception:
                        st.video(out_vid)

                # Defect Events
                st.markdown("#### 📝 Detected Defect Events")
                ev_df = res["events_df"]
                if not ev_df.empty:
                    st.dataframe(ev_df.drop(columns=["Snapshot"], errors="ignore"), use_container_width=True, hide_index=True)
                    csv_data = ev_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Defect Report (CSV)", csv_data, f"defect_report_{cleaned_name}.csv", "text/csv")
                else:
                    st.info("No defects detected above the threshold.")

# ─────────────────── Tab 2: Live Camera & YOLO Pipeline ──────────────────────
with tab_live:
    # Fragment to auto-refresh live camera feed and alerts smoothly
    @st.fragment(run_every="0.25s")
    def render_live_inspection():
        s = runtime.snapshot()
        run_state = s["running"]
        stt = s["stats"]
        al = s["alert"]
        cts = s["class_counts"]
        t_def = sum(cts.values())
        
        # 1. Alert Banner
        if al["active"]:
            info = al["info"]
            cls_name = info.get("class", "DEFECT").upper()
            c_val = info.get("confidence", 0.0) * 100
            st.markdown(f"""
            <div class="alert-banner-danger">
                🚨 CRITICAL DEFECT DETECTED: {cls_name} ({c_val:.1f}%) — STOP MOVING FABRIC 🚨
            </div>
            """, unsafe_allow_html=True)
        elif run_state:
            st.markdown("""
            <div class="alert-banner-normal">
                🟢 SCANNING FABRIC FEED — SYSTEM NORMAL (NO DEFECTS IN CURRENT VIEW)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("System is in standby. Click 'START' in the sidebar to begin live inspection.")
            
        # 2. Defect Counters Row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #f85149;">
                <div class="metric-label">Total Defects</div>
                <div class="metric-val" style="color: #f85149;">{t_def}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #da3633;">
                <div class="metric-label">Holes</div>
                <div class="metric-val">{cts.get('hole', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #d29922;">
                <div class="metric-label">Stains</div>
                <div class="metric-val">{cts.get('stain', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #e36209;">
                <div class="metric-label">Broken Threads</div>
                <div class="metric-val">{cts.get('broken_thread', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #bc8cff;">
                <div class="metric-label">Irregular Yarn</div>
                <div class="metric-val">{cts.get('irregular_yarn', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # 3. Main Live Stream + Live Performance
        col_video, col_telemetry = st.columns([7, 3])
        
        with col_video:
            st.markdown("#### Live Inspection Footage")
            frame = s["frame"]
            if frame is not None and frame.size > 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, channels="RGB", use_container_width=True)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "WAITING FOR CAMERA STREAM...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                st.image(placeholder, channels="BGR", use_container_width=True)
                
        with col_telemetry:
            st.markdown("#### Real-Time Telemetry")
            
            cam_fps = stt.get("camera_fps", 0.0)
            ai_fps = stt.get("inference_fps", 0.0)
            lat_ms = stt.get("inference_ms", 0.0)
            frames_proc = stt.get("frames_processed", 0)
            
            st.metric("Inference Latency", f"{lat_ms:.1f} ms", delta=f"{ai_fps:.1f} AI FPS")
            st.metric("Camera Capture Speed", f"{cam_fps:.1f} FPS")
            st.metric("Frames Processed", f"{frames_proc:,}")
            
            # Live Probability Distribution (for Classifiers)
            probs = s.get("class_probs", {})
            if probs:
                st.markdown("---")
                st.markdown("##### Live Class Probabilities")
                for c_name, p_val in sorted(probs.items(), key=lambda x: -x[1]):
                    col_bar, col_lbl = st.columns([7, 3])
                    with col_bar:
                        st.progress(min(1.0, max(0.0, p_val)))
                    with col_lbl:
                        st.caption(f"{c_name}: **{p_val*100:.1f}%**")

    render_live_inspection()

# ─────────────────── Tab 2: Model Evaluation & Benchmarks ────────────────────
with tab_eval:
    st.markdown("### 📊 Model Evaluation & Benchmarking Study")
    st.markdown("""
    To validate model efficacy rigorously for academic and industrial requirements, we evaluated our primary
    architecture against both lighter edge alternatives and previous generation models on **558 held-out test samples**.
    """)
    
    # Benchmark table
    benchmark_file = Path("logs/benchmark_results.txt")
    if benchmark_file.exists():
        st.code(benchmark_file.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Running benchmarks will populate this section.")
        
    st.markdown("---")
    st.markdown("### 🏆 Architecture Trade-off Analysis")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #3fb950;">
            <h4 style="color: #3fb950; margin: 0;">🥇 YOLO11n-cls (Nano)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Best Overall Model</div>
            <p>• <b>Top-1 Acc: 89.96%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: <b>1.54M</b> (72% smaller)<br>
            • Model Size: <b>3.0 MB</b><br>
            • Latency: <b>1.58 ms / frame</b></p>
            <span style="color: #3fb950; font-size: 12px;"><b>Verdict:</b> Optimal for high-speed edge video feeds. Avoids weave overfitting.</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #58a6ff;">
            <h4 style="color: #58a6ff; margin: 0;">🥈 YOLO11s-cls (Small)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">High Capacity Baseline</div>
            <p>• <b>Top-1 Acc: 88.89%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: 5.45M<br>
            • Model Size: 10.5 MB<br>
            • Latency: 2.17 ms / frame</p>
            <span style="color: #58a6ff; font-size: 12px;"><b>Verdict:</b> Rich feature representations; solid high-resolution performance.</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #8b949e;">
            <h4 style="color: #8b949e; margin: 0;">🥉 YOLOv8s-cls (YOLOv8)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Prior Generation Benchmark</div>
            <p>• <b>Top-1 Acc: 87.63%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: 5.09M<br>
            • Model Size: 9.8 MB<br>
            • Latency: 1.54 ms / frame</p>
            <span style="color: #8b949e; font-size: 12px;"><b>Verdict:</b> YOLO11 delivers a +1.3% to +2.3% accuracy increase over YOLOv8.</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────── Tab 3: Defect Snapshots & History ───────────────────────
with tab_history:
    st.markdown("### 📸 Live Defect Snapshot Gallery & Event History")
    
    events_df = runtime.logger.events_df(runtime.session_id)
    
    if not events_df.empty:
        # Gallery of snapshots
        snaps_with_img = events_df[events_df["snapshot_path"].str.len() > 0]
        if not snaps_with_img.empty:
            st.markdown("#### Captured Defect Evidence")
            gallery_cols = st.columns(4)
            for idx, (_, row) in enumerate(snaps_with_img.head(8).iterrows()):
                col = gallery_cols[idx % 4]
                img_p = row["snapshot_path"]
                if Path(img_p).exists():
                    col.image(img_p, caption=f"#{row['event_id']} {row['defect_class']} ({row['confidence']:.2f})")
        
        st.markdown("---")
        st.markdown("#### Event Audit Log")
        display_cols = ["event_id", "defect_class", "confidence", "first_seen", "frames_seen", "snapshot_path"]
        st.dataframe(events_df[display_cols], use_container_width=True)
    else:
        st.info("No defect events recorded in current session yet.")

# ─────────────────── Tab 4: Session Records & Export ─────────────────────────
with tab_sessions:
    st.markdown("### 📁 Historical Inspection Sessions")
    
    sess_df = runtime.logger.sessions_df()
    if not sess_df.empty:
        st.dataframe(sess_df, use_container_width=True)
        
        # CSV Export
        all_events_df = runtime.logger.events_df()
        if not all_events_df.empty:
            csv_data = all_events_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Defects CSV Report",
                data=csv_data,
                file_name=f"fabric_defects_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.info("No saved inspection sessions found in database.")
