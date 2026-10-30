"""
Dashboard runtime glue layer.
Manages the inspection pipeline, database logger, real-time alert state,
audio notifications, and state snapshots for Streamlit polling.
"""
import copy
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from defect_logging.defect_logger import DefectLogger
from streaming.camera import OpenCVCamera
from vision.detector import FabricDetector
from vision.pipeline import InspectionPipeline, load_config
from vision.tracker import DefectEventTracker
from .mocks import SyntheticFabricCamera, BlobDetector

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class DashboardRuntime:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.cfg = load_config(config_path) if Path(config_path).exists() else {}
        self.logger = DefectLogger(
            db_path=self.cfg.get("logging", {}).get("db_path", "logs/inspection.db"),
            snapshot_dir=self.cfg.get("logging", {}).get("snapshot_dir", "logs/snapshots")
        )
        self.pipeline: Optional[InspectionPipeline] = None
        self.session_id: Optional[str] = None
        self._lock = threading.Lock()
        
        # Alert state
        self._active_alert = False
        self._alert_info = {}
        self._alert_timestamp = 0.0
        self._alert_duration = float(self.cfg.get("alerts", {}).get("banner_seconds", 3.0))
        self._sound_enabled = bool(self.cfg.get("alerts", {}).get("beep", True))
        
        # Defect tracking stats
        self._class_counts: Dict[str, int] = {
            "hole": 0,
            "stain": 0,
            "broken_thread": 0,
            "irregular_yarn": 0
        }
        self._recent_events: List[dict] = []
        self._last_error = ""

    def start_session(
        self,
        fabric_id: str = "Roll-1",
        fabric_length_m: Optional[float] = 10.0,
        source: str = "0",
        weights: str = "models/fabric_classifier_best.pt",
        mock_mode: bool = False,
        conf: float = 0.50,
        lighting: str = "clahe",
        beep: bool = True
    ):
        with self._lock:
            if self.pipeline and self.pipeline.running:
                return self.session_id

            self._sound_enabled = beep
            self._class_counts = {"hole": 0, "stain": 0, "broken_thread": 0, "irregular_yarn": 0}
            self._recent_events = []
            self._active_alert = False
            self._alert_info = {}
            self._last_error = ""

            # Start database session
            self.session_id = self.logger.start_session(
                fabric_id=fabric_id,
                fabric_length_m=fabric_length_m,
                notes=f"weights={Path(weights).name}, mock={mock_mode}, lighting={lighting}"
            )

            try:
                # 1. Setup Camera
                if mock_mode or source.lower() == "synthetic":
                    camera = SyntheticFabricCamera(width=640, height=480, fps=25)
                else:
                    cam_src = int(source) if source.isdigit() else source
                    camera = OpenCVCamera(
                        source=cam_src,
                        width=self.cfg.get("camera", {}).get("width", 640),
                        height=self.cfg.get("camera", {}).get("height", 480),
                        fps=self.cfg.get("camera", {}).get("fps", 30)
                    )

                # 2. Setup Detector
                if mock_mode and not Path(weights).exists():
                    detector = BlobDetector(conf=conf)
                else:
                    # Fallback to alternative trained weights if specified file missing
                    actual_weights = weights
                    if not Path(actual_weights).exists():
                        fallbacks = [
                            "models/fabric_classifier_best.pt",
                            "models/yolo11n_classifier_best.pt",
                            "runs/classify/runs/cls_yolo11n/weights/best.pt",
                            "runs/cls/weights/best.pt",
                            "yolo11n-cls.pt"
                        ]
                        for fb in fallbacks:
                            if Path(fb).exists():
                                actual_weights = fb
                                break
                    detector = FabricDetector(
                        weights=actual_weights,
                        conf=conf,
                        imgsz=224,
                        lighting_method=lighting if lighting != "none" else None,
                        to_gray=True
                    )

                # 3. Setup Tracker
                tr_cfg = self.cfg.get("tracker", {})
                tracker = DefectEventTracker(
                    min_confidence=tr_cfg.get("min_confidence", 0.45),
                    confirm_confidence=tr_cfg.get("confirm_confidence", 0.65),
                    confirm_frames=tr_cfg.get("confirm_frames", 2),
                    max_missed=tr_cfg.get("max_missed", 10),
                    iou_threshold=tr_cfg.get("iou_threshold", 0.20),
                    max_center_dist=tr_cfg.get("max_center_dist", 0.20)
                )

                # 4. Assemble Pipeline with greyscale & lighting preprocessing
                prep_cfg = copy.deepcopy(self.cfg.get("preprocessing", {}))
                if lighting and lighting != "none":
                    prep_cfg["lighting_correction"] = True
                    prep_cfg["method"] = lighting
                    prep_cfg["to_gray"] = True

                self.pipeline = InspectionPipeline(
                    camera=camera,
                    detector=detector,
                    tracker=tracker,
                    on_event=self._on_event_confirmed,
                    on_event_closed=self._on_event_closed,
                    draw=True,
                    preprocessing=prep_cfg
                )
                self.pipeline.start()
            except Exception as e:
                self._last_error = str(e)
                if self.pipeline:
                    self.pipeline.stop()
                    self.pipeline = None
                raise

            return self.session_id

    def stop_session(self):
        with self._lock:
            if not self.pipeline:
                return
            frames_processed = 0
            if self.pipeline:
                stats = self.pipeline.stats()
                frames_processed = stats.get("frames_processed", 0)
                self.pipeline.stop()
                self.pipeline = None
            if self.session_id:
                self.logger.end_session(self.session_id, frames_processed=frames_processed)
            self._active_alert = False

    def _on_event_confirmed(self, event: dict):
        """Called when a defect is confirmed by the tracker."""
        now = time.time()
        d_class = event.get("class", "defect")
        conf = float(event.get("confidence", 0.0))

        with self._lock:
            self._active_alert = True
            self._alert_timestamp = now
            self._alert_info = {
                "class": d_class,
                "confidence": conf,
                "event_id": event.get("event_id", 0),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            if d_class in self._class_counts:
                self._class_counts[d_class] += 1
            else:
                self._class_counts[d_class] = 1

            self._recent_events.insert(0, copy.deepcopy(event))
            if len(self._recent_events) > 50:
                self._recent_events.pop()

            # Save snapshot to disk and log to DB
            frame = None
            if self.pipeline:
                lat = self.pipeline.latest()
                if lat and lat.get("frame") is not None:
                    frame = lat["frame"]
            snap_path = self.logger.log_event(self.session_id, event, frame=frame)
            event["snapshot_path"] = snap_path

        # Trigger audio alert in separate thread
        if self._sound_enabled and HAS_WINSOUND:
            threading.Thread(target=self._play_beep, daemon=True).start()

    def _play_beep(self):
        try:
            winsound.Beep(1200, 250)
        except Exception:
            pass

    def _on_event_closed(self, event: dict):
        """Called when a defect leaves the view."""
        if self.session_id:
            self.logger.close_event(self.session_id, event)

    def snapshot(self) -> dict:
        """Fetch current frame, performance stats, alerts, and counts for dashboard render."""
        with self._lock:
            is_running = self.pipeline is not None and self.pipeline.running
            pipe_latest = self.pipeline.latest() if is_running else None
            stats = self.pipeline.stats() if is_running else {}
            
            # Check alert expiration
            now = time.time()
            if self._active_alert and (now - self._alert_timestamp) > self._alert_duration:
                self._active_alert = False

            # Model probabilities if classifier is used
            class_probs = {}
            if is_running and hasattr(self.pipeline, "detector") and hasattr(self.pipeline.detector, "last_probs"):
                class_probs = getattr(self.pipeline.detector, "last_probs", {})

            return {
                "running": is_running,
                "session_id": self.session_id,
                "frame": pipe_latest["frame"] if pipe_latest else None,
                "stats": stats,
                "alert": {
                    "active": self._active_alert,
                    "info": dict(self._alert_info)
                },
                "class_counts": dict(self._class_counts),
                "class_probs": class_probs,
                "recent_events": list(self._recent_events),
                "error": self._last_error
            }


# Singleton instance accessor
_runtime_instance: Optional[DashboardRuntime] = None

def get_runtime() -> DashboardRuntime:
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = DashboardRuntime()
    return _runtime_instance
