"""
FabricDetector - the ONLY thing other modules need from the ML side (team contract unchanged).
    det = FabricDetector("models/fabric_defect.pt", conf=0.25, imgsz=640)
    det.warmup()
    detections = det.detect(frame_bgr)   # list of {"class","class_id","confidence","bbox":[x1,y1,x2,y2]}

Changes vs. the original guide:
  * works with BOTH normal YOLO boxes and oriented boxes (OBB). For OBB models the returned "bbox" is the axis-aligned
    envelope of the rotated box, so the tracker / dashboard need no changes. The rotated corners are also returned in
    "obb_polygon" (4 x [x, y]) if you want to draw them.
  * to_gray=True (default): the models were TRAINED ON GRAYSCALE images, so frames are converted to gray (still 3-channel)
    before inference. Use to_gray=False only for a model you trained on colour.
Works with .pt, .onnx or .engine weights. Knows nothing about cameras or dashboards.
"""
from pathlib import Path

import cv2
import numpy as np


class FabricDetector:
    def __init__(self, weights="models/fabric_classifier_best.pt", conf=0.25, imgsz=None, device=None, half=True, to_gray=True, lighting_method=None):
        import torch
        from ultralytics import YOLO

        resolved_weights = Path(str(weights))
        repo_root = Path(__file__).resolve().parents[1]
        if not resolved_weights.exists() and not str(weights).startswith(("yolo", "http")):
            alt = repo_root / weights
            if alt.exists():
                resolved_weights = alt
            elif str(weights) in ("models/fabric_defect.pt", "fabric_defect.pt", "models/fabric_classifier_best.pt"):
                for cand in [
                    repo_root / "models" / "fabric_classifier_best.pt",
                    repo_root / "models" / "yolo11n_classifier_best.pt",
                    Path("models/fabric_classifier_best.pt"),
                    Path("models/yolo11n_classifier_best.pt"),
                ]:
                    if cand.exists():
                        resolved_weights = cand
                        break

        if not resolved_weights.exists() and not str(weights).startswith(("yolo", "http")):
            raise FileNotFoundError(f"Model weights not found: {weights}")

        self.model = YOLO(str(resolved_weights))
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.half = bool(half) and self.device != "cpu"
        self.conf = float(conf)
        if imgsz is not None:
            self.imgsz = int(imgsz)
        elif getattr(self.model, "task", None) == "classify":
            self.imgsz = 224
        else:
            self.imgsz = 640
        self.to_gray = bool(to_gray)
        self.lighting_method = lighting_method
        self.names = self.model.names            # {0: "hole", 1: "stain", 2: "broken_thread", 3: "irregular_yarn"}
        self.last_probs = {}

    def warmup(self, n=3):
        dummy = np.full((self.imgsz, self.imgsz, 3), 127, np.uint8)
        for _ in range(n):
            self.detect(dummy)

    def _prep(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return frame
        if self.lighting_method:
            from .preprocessing import correct_lighting
            return correct_lighting(frame, method=self.lighting_method, to_gray=self.to_gray)
        if not self.to_gray:
            return frame
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            g = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3:
            g = frame[:, :, 0]
        else:
            g = frame
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def detect(self, frame):
        """frame: BGR uint8 ndarray (OpenCV image). Returns list of detection dicts."""
        predict_kwargs = {
            "conf": self.conf,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            predict_kwargs["half"] = True
        r = self.model.predict(self._prep(frame), **predict_kwargs)[0]
        out = []
        # Support classification models (YOLO11-cls, YOLOv8-cls)
        probs = getattr(r, "probs", None)
        if probs is not None:
            self.last_probs = {self.names[i]: float(probs.data[i]) for i in range(len(self.names))}
            top1_id = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_name = self.names[top1_id]
            if top1_name != "non_defective" and top1_conf >= self.conf:
                h, w = frame.shape[:2]
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                out.append({
                    "class": top1_name,
                    "class_id": top1_id,
                    "confidence": top1_conf,
                    "bbox": [margin_x, margin_y, w - margin_x, h - margin_y]
                })
            return out

        obb = getattr(r, "obb", None)
        if obb is not None and len(obb) > 0:                                  # oriented-box model
            env = obb.xyxy.cpu().numpy()
            poly = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            clss = obb.cls.cpu().numpy().astype(int)
            for b, p, s, c in zip(env, poly, confs, clss):
                out.append({"class": self.names[c], "class_id": int(c), "confidence": float(s),
                            "bbox": [int(round(v)) for v in b],
                            "obb_polygon": [[int(round(x)), int(round(y))] for x, y in p]})
            return out
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        return [{"class": self.names[c], "class_id": int(c), "confidence": float(s),
                 "bbox": [int(round(v)) for v in b]} for b, s, c in zip(xyxy, confs, clss)]
