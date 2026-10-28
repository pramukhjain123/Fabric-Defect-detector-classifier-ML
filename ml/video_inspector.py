"""
Classical Machine Learning Video Inspection Engine.
Processes uploaded fabric videos (simulating line-scan cameras or moving fabric),
localizes candidate anomalies, extracts 64 handcrafted features, classifies them
with Classical ML (Random Forest, SVM, or Gradient Boosting), tracks defects across frames,
and produces annotated video and comparative metrics.
"""
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
import pandas as pd

from ml.features import extract_patch_features, DEFECT_CLASSES, ID_TO_CLASS, CLASS_TO_ID
from vision.postprocessing import CLASS_COLORS, color_for
from vision.preprocessing import correct_lighting
from vision.tracker import DefectEventTracker


MODEL_PATHS = {
    "Random Forest": "models/ml_random_forest.joblib",
    "Support Vector Machine (SVM)": "models/ml_svm.joblib",
    "Gradient Boosting (HistGBM)": "models/ml_gradient_boosting.joblib",
}


def nms_candidate_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Suppresses heavily overlapping candidate bounding boxes to prevent duplicate tracks."""
    if not boxes:
        return []
    # Sort descending by bounding box area
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in sorted_boxes:
        bx1, by1, bw, bh = b
        bx2, by2 = bx1 + bw, by1 + bh
        overlap = False
        for k in kept:
            kx1, ky1, kw, kh = k
            kx2, ky2 = kx1 + kw, ky1 + kh
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + kw * kh - inter
            if union > 0 and (inter / union) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(b)
    return kept


def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], max_dist: int = 45) -> List[Tuple[int, int, int, int]]:
    """Merges adjacent bounding boxes belonging to the same physical defect (e.g. elongated tears or grouped holes)."""
    if not boxes:
        return []
    rects = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
    merged = True
    while merged:
        merged = False
        new_rects = []
        skip = set()
        for i in range(len(rects)):
            if i in skip:
                continue
            r1 = rects[i]
            for j in range(i + 1, len(rects)):
                if j in skip:
                    continue
                r2 = rects[j]
                x_dist = max(0, max(r1[0], r2[0]) - min(r1[2], r2[2]))
                y_dist = max(0, max(r1[1], r2[1]) - min(r1[3], r2[3]))
                if x_dist <= max_dist and y_dist <= max_dist:
                    r1 = [min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3])]
                    skip.add(j)
                    merged = True
            new_rects.append(r1)
        rects = new_rects
    return [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects]


def deduplicate_closed_events(events: List[Dict], max_center_dist: float = 220.0, max_frame_gap: int = 60) -> List[Dict]:
    """Merges duplicate detections and tracks of the same physical defect across time and camera motion."""
    if not events:
        return []
    sorted_evs = sorted(events, key=lambda e: e["first_frame"])
    merged_evs = []

    for ev in sorted_evs:
        matched = False
        bx1, by1, bx2, by2 = ev["bbox"]
        cx1, cy1 = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        for m in merged_evs:
            mbx1, mby1, mbx2, mby2 = m["bbox"]
            mcx1, mcy1 = (mbx1 + mbx2) / 2.0, (mby1 + mby2) / 2.0
            dist = math.hypot(cx1 - mcx1, cy1 - mcy1)

            time_close = (ev["first_frame"] <= m["last_frame"] + max_frame_gap) and (m["first_frame"] <= ev["last_frame"] + max_frame_gap)
            spatial_close = dist <= max_center_dist

            if spatial_close and time_close:
                m["first_frame"] = min(m["first_frame"], ev["first_frame"])
                m["last_frame"] = max(m["last_frame"], ev["last_frame"])
                m["frames_seen"] = m.get("frames_seen", 1) + ev.get("frames_seen", 1)
                # Keep highest confidence prediction and prioritize hole if detected
                if ev.get("class") == "hole" or (ev.get("confidence", 0.0) > m.get("confidence", 0.0) and m.get("class") != "hole"):
                    m["confidence"] = ev.get("confidence", m.get("confidence"))
                    m["class"] = ev.get("class", m.get("class"))
                m["bbox"] = [
                    min(mbx1, bx1), min(mby1, by1),
                    max(mbx2, bx2), max(mby2, by2)
                ]
                matched = True
                break

        if not matched:
            merged_evs.append(dict(ev))

    for idx, me in enumerate(merged_evs, 1):
        me["event_id"] = idx
    return merged_evs


class ClassicalMLVideoInspector:
    def __init__(self, model_name: str = "Random Forest"):
        self.model_name = model_name
        self.model_path = MODEL_PATHS.get(model_name, "models/ml_random_forest.joblib")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Run ml/train_models.py first.")
        
        self.model = joblib.load(self.model_path)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.use_scaled = (model_name == "Support Vector Machine (SVM)")
        self.scaler = joblib.load("models/ml_scaler.joblib") if Path("models/ml_scaler.joblib").exists() else None

    def predict_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        """Extract features and classify a single fabric patch."""
        feats = extract_patch_features(patch).reshape(1, -1)
        if self.use_scaled and self.scaler is not None:
            feats = self.scaler.transform(feats)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feats)[0]
            top_id = int(np.argmax(probs))
            conf = float(probs[top_id])
        else:
            top_id = int(self.model.predict(feats)[0])
            conf = 1.0

        class_name = ID_TO_CLASS.get(top_id, "non_defective")
        return class_name, conf

    def detect_candidate_regions(self, gray_frame: np.ndarray, min_area: int = 100, max_area: int = 30000) -> List[Tuple[int, int, int, int]]:
        """Find candidate defect bounding boxes using adaptive thresholding and texture deviation.
        
        Applies border margins to eliminate edge vignette noise and NMS to deduplicate boxes.
        """
        h, w = gray_frame.shape[:2]
        # Illumination subtraction: smooth background estimate
        bg = cv2.GaussianBlur(gray_frame, (31, 31), 0)
        diff = cv2.absdiff(gray_frame, bg)

        # Adaptive thresholding
        mean_d, std_d = cv2.meanStdDev(diff)
        thresh_val = int(mean_d[0][0] + 1.8 * std_d[0][0] + 12)
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        border_top = max(50, int(h * 0.07))
        border_bottom = max(80, int(h * 0.10))
        border_x = max(45, int(w * 0.10))

        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                # Ignore edge artifacts / table borders / raw cloth boundary fringes
                if y < border_top or (y + bh) > (h - border_bottom) or x < border_x or (x + bw) > (w - border_x):
                    continue

                # Expand box slightly to include weave context
                pad = 12
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw + pad)
                y2 = min(h, y + bh + pad)
                candidates.append((x1, y1, x2 - x1, y2 - y1))

        # Merge adjacent disjoint contours belonging to the same tear/defect
        merged = merge_nearby_boxes(candidates, max_dist=45)
        # Suppress overlapping candidate boxes
        deduped = nms_candidate_boxes(merged, iou_thresh=0.20)
        return deduped[:6]

    def inspect_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        conf_threshold: float = 0.50,
        snapshot_dir: str = "logs/ml_video_snapshots",
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """Run complete Classical ML inspection on an uploaded video file.
        
        Args:
            video_path: Path to input video file (.mp4, .avi, etc.).
            output_video_path: Optional path to write annotated output video.
            conf_threshold: Confidence threshold for candidate defects.
            snapshot_dir: Directory to save thumbnail snapshots of defects.
            progress_cb: Optional callback(fraction: float, message: str).
            
        Returns:
            Dict containing events, events_df, stats, and output paths.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        raw_video_path = None
        if output_video_path:
            raw_video_path = str(output_video_path) + ".raw.mp4"
            Path(raw_video_path).parent.mkdir(parents=True, exist_ok=True)
            # Use mp4v for initial raw encoding
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_video_path, fourcc, video_fps, (width, height))

        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        tracker = DefectEventTracker(
            min_confidence=conf_threshold * 0.85,
            confirm_confidence=conf_threshold,
            confirm_frames=3,
            max_missed=60,
            iou_threshold=0.15,
            max_center_dist=0.25
        )

        frame_idx = 0
        total_inferences = 0
        frames_with_defect = 0
        all_new_events = []
        all_closed_events = []
        snapshots_map = {}
        frame_stride = 2 if total_frames > 200 else 1

        t_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # 1. Preprocessing & candidate localization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidates = self.detect_candidate_regions(gray)

            raw_detections = []
            if candidates:
                patches = [gray[cy:cy + ch, cx:cx + cw] for (cx, cy, cw, ch) in candidates]
                feats_list = [extract_patch_features(p) for p in patches]
                feats_batch = np.array(feats_list, dtype=np.float32)
                if self.use_scaled and self.scaler is not None:
                    feats_batch = self.scaler.transform(feats_batch)

                if hasattr(self.model, "predict_proba"):
                    probs_batch = self.model.predict_proba(feats_batch)
                    top_ids = np.argmax(probs_batch, axis=1)
                    confs = np.max(probs_batch, axis=1)
                else:
                    top_ids = self.model.predict(feats_batch)
                    confs = np.ones(len(top_ids), dtype=np.float32)

                for (cx, cy, cw, ch), top_id, conf in zip(candidates, top_ids, confs):
                    total_inferences += 1
                    cls_name = ID_TO_CLASS.get(int(top_id), "non_defective")
                    if cls_name != "non_defective" and float(conf) >= conf_threshold:
                        raw_detections.append({
                            "class": cls_name,
                            "class_id": CLASS_TO_ID.get(cls_name, 0),
                            "confidence": float(conf),
                            "bbox": [cx, cy, cx + cw, cy + ch]
                        })

            if raw_detections:
                frames_with_defect += 1

            # 3. Defect Event Tracking across frames
            new_events, closed_events = tracker.update(raw_detections, frame_idx, frame)
            all_new_events.extend(new_events)
            all_closed_events.extend(closed_events)

            # Save snapshot thumbnails for confirmed events
            for ev in new_events:
                ev_id = ev["event_id"]
                x1, y1, x2, y2 = ev["bbox"]
                crop = frame[max(0, y1 - 10):min(height, y2 + 10), max(0, x1 - 10):min(width, x2 + 10)]
                if crop.size > 0:
                    snap_path = Path(snapshot_dir) / f"defect_event_{ev_id}_{ev['class']}.jpg"
                    cv2.imwrite(str(snap_path), crop)
                    snapshots_map[ev_id] = str(snap_path)

            active_tracks = tracker.active_tracks()

            # 4. Draw annotations for video output
            if writer:
                annotated = frame.copy()
                # Draw top status banner
                cv2.rectangle(annotated, (0, 0), (width, 36), (20, 20, 20), -1)
                status_txt = f"Model: {self.model_name} | Frame: {frame_idx}/{total_frames} | Confirmed Defects: {tracker.total_events}"
                cv2.putText(annotated, status_txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

                # Draw raw candidate detections (thin lines)
                for d in raw_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    c = color_for(d["class"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 1)

                # Draw confirmed tracked defects (thick prominent box + tag)
                for tr in active_tracks:
                    if tr.get("confirmed"):
                        x1, y1, x2, y2 = tr["bbox"]
                        c = color_for(tr["class"])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 3)
                        tag = f"#{tr['event_id']} {tr['class'].upper()} ({tr.get('confidence', 0.0):.2f})"
                        cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + 180, y1), c, -1)
                        cv2.putText(annotated, tag, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                writer.write(np.ascontiguousarray(annotated))

            frame_idx += 1
            if progress_cb and frame_idx % 5 == 0:
                progress_cb(frame_idx / total_frames, f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        if writer:
            writer.release()
            writer = None
            time.sleep(0.2)
            # Transcode raw output video to HTML5 browser-compatible H.264
            if output_video_path and raw_video_path and Path(raw_video_path).exists():
                try:
                    import subprocess
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", raw_video_path,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-preset", "veryfast",
                        str(output_video_path)
                    ]
                    proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if proc.returncode != 0 or not Path(output_video_path).exists():
                        import shutil
                        shutil.copy2(raw_video_path, output_video_path)
                    try:
                        time.sleep(0.1)
                        Path(raw_video_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception:
                    import shutil
                    shutil.copy2(raw_video_path, output_video_path)

        # Flush any remaining open tracks
        remaining_closed = tracker.flush()
        all_closed_events.extend(remaining_closed)

        # Spatio-temporal event deduplication: merge duplicate frame tracks of the same physical defect
        final_events = deduplicate_closed_events(all_closed_events)

        total_time = time.perf_counter() - t_start
        fps = frame_idx / max(0.001, total_time)

        # Build events DataFrame
        event_rows = []
        for ev in final_events:
            ev_id = ev["event_id"]
            first_f = ev["first_frame"]
            last_f = ev["last_frame"]
            duration_s = (last_f - first_f + 1) / video_fps
            start_timestamp = f"{int(first_f / video_fps // 60):02d}:{int(first_f / video_fps % 60):02d}"
            event_rows.append({
                "Defect ID": f"#{ev_id}",
                "Defect Class": ev["class"],
                "Confidence": round(float(ev.get("confidence", 0.0)), 3),
                "Timestamp": start_timestamp,
                "Start Frame": first_f,
                "End Frame": last_f,
                "Duration (s)": round(duration_s, 2),
                "Snapshot": snapshots_map.get(ev_id, "")
            })

        events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame(
            columns=["Defect ID", "Defect Class", "Confidence", "Timestamp", "Start Frame", "End Frame", "Duration (s)", "Snapshot"]
        )

        class_counts = {cls: 0 for cls in DEFECT_CLASSES if cls != "non_defective"}
        for ev in final_events:
            if ev["class"] in class_counts:
                class_counts[ev["class"]] += 1

        stats = {
            "model_name": self.model_name,
            "total_frames": frame_idx,
            "video_duration_sec": round(frame_idx / video_fps, 2),
            "video_fps": video_fps,
            "processing_time_sec": round(total_time, 2),
            "processing_fps": round(fps, 1),
            "average_ms_per_frame": round((total_time / max(1, frame_idx)) * 1000.0, 1),
            "total_candidate_inferences": total_inferences,
            "frames_with_defect": frames_with_defect,
            "total_unique_defects": len(final_events),
            "defect_counts": class_counts,
            "output_video_path": str(output_video_path) if output_video_path else None
        }

        return {
            "events_df": events_df,
            "events": final_events,
            "stats": stats,
            "output_video_path": output_video_path
        }


def run_comparative_video_study(
    video_path: str,
    conf_threshold: float = 0.50,
    progress_cb: Optional[Callable[[float, str], None]] = None
) -> Dict:
    """Run the uploaded video through ALL 3 Classical ML models and return comparative study results."""
    models = ["Random Forest", "Support Vector Machine (SVM)", "Gradient Boosting (HistGBM)"]
    study_results = {}
    summary_rows = []

    for idx, m_name in enumerate(models):
        if progress_cb:
            progress_cb(idx / len(models), f"Running Comparative Analysis with {m_name}...")

        inspector = ClassicalMLVideoInspector(model_name=m_name)
        out_vid = f"logs/annotated_{m_name.replace(' ', '_').lower()}.mp4"
        res = inspector.inspect_video(
            video_path=video_path,
            output_video_path=out_vid,
            conf_threshold=conf_threshold
        )
        study_results[m_name] = res
        s = res["stats"]
        summary_rows.append({
            "ML Model": m_name,
            "Total Defects Found": s["total_unique_defects"],
            "Holes": s["defect_counts"].get("hole", 0),
            "Stains": s["defect_counts"].get("stain", 0),
            "Broken Threads": s["defect_counts"].get("broken_thread", 0),
            "Irregular Yarn": s["defect_counts"].get("irregular_yarn", 0),
            "Processing Speed (FPS)": s["processing_fps"],
            "Avg Frame Latency (ms)": s["average_ms_per_frame"],
            "Total Time (s)": s["processing_time_sec"]
        })

    comparative_df = pd.DataFrame(summary_rows)
    return {
        "comparative_df": comparative_df,
        "model_results": study_results
    }
