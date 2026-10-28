"""
Video Self-Learning & Training Engine for Classical Machine Learning.
Ingests fabric videos (real camera recordings or synthetic conveyor feeds),
mines candidate and confirmed defect patches using teacher vision detectors,
extracts 64 handcrafted texture/gradient/morphology features, augments the data,
retrains 3 Classical ML models (Random Forest, SVM, HistGBM), and updates models/metrics.
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.features import (
    CLASS_TO_ID,
    DEFECT_CLASSES,
    ID_TO_CLASS,
    extract_patch_features,
    get_feature_names,
)
from ml.train_models import augment_crop_variants, create_synthetic_fabric_patch
from vision.detector import FabricDetector

PATCH_DUMP_DIR = Path("logs/mined_video_patches")


def augment_video_patch(crop: np.ndarray, class_name: str, target_size: int = 64) -> List[Tuple[np.ndarray, int]]:
    """Generate 4 key orientation variants of a mined patch for rapid, robust ML training."""
    cid = CLASS_TO_ID.get(class_name, 3)
    variants = []
    if crop is None or crop.size == 0:
        return variants
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    p0 = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    variants.append((extract_patch_features(p0), cid))

    # 90 deg rotation
    p90 = cv2.rotate(p0, cv2.ROTATE_90_CLOCKWISE)
    variants.append((extract_patch_features(p90), cid))

    # Horizontal flip
    pfl = cv2.flip(p0, 1)
    variants.append((extract_patch_features(pfl), cid))

    # Contrast enhanced variant
    p_adj = np.clip(p0.astype(np.float32) * 1.08 + 5, 0, 255).astype(np.uint8)
    variants.append((extract_patch_features(p_adj), cid))
    return variants


class VideoMLTrainer:
    """Self-training pipeline that learns from fabric videos and updates Classical ML models."""

    def __init__(self, detector_conf: float = 0.25):
        self.detector_conf = detector_conf
        self._detector: Optional[FabricDetector] = None

    @property
    def detector(self) -> FabricDetector:
        if self._detector is None:
            self._detector = FabricDetector(conf=self.detector_conf)
        return self._detector

    def mine_patches_from_video(
        self,
        video_path: str,
        max_frames: int = 500,
        frame_stride: int = 4,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict]:
        """Scan a fabric video, detect defects, crop surrounding fabric patches, and extract clean patches."""
        PATCH_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video at: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        mined_records = []
        frame_idx = 0
        clean_patches_mined = 0
        defect_patches_mined = 0
        video_stem = Path(video_path).stem[:16].replace(" ", "_")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame_idx >= max_frames:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            if progress_cb and frame_idx % 20 == 0:
                pct = min(0.9, frame_idx / max(1, min(total_frames, max_frames)))
                progress_cb(pct, f"Mining video frame {frame_idx}/{min(total_frames, max_frames)}...")

            # 1. Detect defects using teacher model
            detections = self.detector.detect(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detections:
                for d_idx, det in enumerate(detections):
                    cls_name = det["class"]
                    conf = float(det["confidence"])
                    if cls_name == "non_defective":
                        continue

                    x1, y1, x2, y2 = det["bbox"]
                    bw, bh = x2 - x1, y2 - y1

                    # Add padding around defect to capture surrounding textile weave context
                    pad = int(max(12, min(bw, bh) * 0.25))
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(width, x2 + pad)
                    cy2 = min(height, y2 + pad)

                    crop = gray_frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                        patch_filename = f"{video_stem}_f{frame_idx}_d{d_idx}_{cls_name}.jpg"
                        patch_path = PATCH_DUMP_DIR / patch_filename
                        cv2.imwrite(str(patch_path), crop)

                        mined_records.append({
                            "class": cls_name,
                            "confidence": conf,
                            "frame_idx": frame_idx,
                            "timestamp_s": round(frame_idx / fps, 2),
                            "crop": crop,
                            "patch_path": str(patch_path),
                            "bbox": [x1, y1, x2, y2],
                            "is_defect": True
                        })
                        defect_patches_mined += 1

            # 2. Mine clean background textile patches (if no defects nearby or sparsely sampled)
            if clean_patches_mined < (defect_patches_mined + 4) * 2:
                # Sample corners/centers far from borders
                grid_y = [height // 4, height // 2, 3 * height // 4]
                grid_x = [width // 4, width // 2, 3 * width // 4]
                for gy in grid_y:
                    for gx in grid_x:
                        # Check collision with any detections
                        collides = False
                        for det in detections:
                            bx1, by1, bx2, by2 = det["bbox"]
                            if (bx1 - 30 <= gx <= bx2 + 30) and (by1 - 30 <= gy <= by2 + 30):
                                collides = True
                                break
                        if not collides and clean_patches_mined < 40:
                            c_crop = gray_frame[max(0, gy - 32):min(height, gy + 32), max(0, gx - 32):min(width, gx + 32)]
                            if c_crop.shape == (64, 64):
                                patch_filename = f"{video_stem}_f{frame_idx}_clean_{clean_patches_mined}.jpg"
                                patch_path = PATCH_DUMP_DIR / patch_filename
                                cv2.imwrite(str(patch_path), c_crop)
                                mined_records.append({
                                    "class": "non_defective",
                                    "confidence": 1.0,
                                    "frame_idx": frame_idx,
                                    "timestamp_s": round(frame_idx / fps, 2),
                                    "crop": c_crop,
                                    "patch_path": str(patch_path),
                                    "bbox": [gx - 32, gy - 32, gx + 32, gy + 32],
                                    "is_defect": False
                                })
                                clean_patches_mined += 1
                                break

            frame_idx += 1

        cap.release()
        print(f"[VideoMLTrainer] Mined {len(mined_records)} patches from {video_path} (Defects: {defect_patches_mined}, Clean: {clean_patches_mined})")
        return mined_records

    def build_dataset_from_mined_patches(
        self,
        mined_records: List[Dict],
        synthetic_base_samples: int = 80,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Augment mined video patches and blend with balanced synthetic baseline to avoid catastrophic forgetting."""
        X_list = []
        y_list = []
        counts = {cls: 0 for cls in DEFECT_CLASSES}

        # 1. Base synthetic weave anchors
        if progress_cb:
            progress_cb(0.1, "Building base weave representations...")
        for class_id, class_name in enumerate(DEFECT_CLASSES):
            for i in range(synthetic_base_samples):
                patch = create_synthetic_fabric_patch(class_name, size=64, seed=class_id * 500 + i)
                feat = extract_patch_features(patch)
                X_list.append(feat)
                y_list.append(class_id)
                counts[class_name] += 1

        # 2. Add real video-mined patches with fast augmentations
        total_rec = len(mined_records)
        # Sample or limit to 50 mined records per class to keep training snappy and perfectly balanced
        per_class_mined = {cls: 0 for cls in DEFECT_CLASSES}
        for idx, rec in enumerate(mined_records):
            cls_name = rec["class"]
            if per_class_mined.get(cls_name, 0) >= 40:
                continue
            crop = rec["crop"]
            cid = CLASS_TO_ID.get(cls_name, 3)

            variants = augment_video_patch(crop, cls_name, target_size=64)
            for feat, _ in variants:
                X_list.append(feat)
                y_list.append(cid)
                counts[cls_name] += 1
            per_class_mined[cls_name] = per_class_mined.get(cls_name, 0) + 1

            if progress_cb and idx % 10 == 0:
                progress_cb(0.1 + 0.3 * (idx / max(1, total_rec)), f"Augmenting mined patch {idx+1}/{total_rec}...")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y, counts

    def train_models_from_videos(
        self,
        video_paths: List[str],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict:
        """Complete workflow: mine defect patches from videos, extract features, retrain 3 ML models, and persist.
        
        Args:
            video_paths: List of file paths to input fabric inspection videos.
            progress_cb: Optional callback(pct, msg).
            
        Returns:
            Dictionary containing metrics before & after, per-model evaluations, and mined sample gallery.
        """
        t0_total = time.perf_counter()
        Path("models").mkdir(parents=True, exist_ok=True)
        all_mined_records = []

        # 1. Mine patches across all provided videos
        num_vids = len(video_paths)
        for v_idx, v_path in enumerate(video_paths):
            if progress_cb:
                progress_cb(
                    0.05 + 0.35 * (v_idx / max(1, num_vids)),
                    f"Mining video {v_idx+1}/{num_vids}: {Path(v_path).name}..."
                )
            records = self.mine_patches_from_video(v_path, progress_cb=progress_cb)
            all_mined_records.extend(records)

        # 2. Build feature matrix X and label vector y
        if progress_cb:
            progress_cb(0.45, f"Assembling dataset from {len(all_mined_records)} mined patches...")
        X, y, class_distribution = self.build_dataset_from_mined_patches(
            all_mined_records, synthetic_base_samples=180, progress_cb=progress_cb
        )

        # 3. Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Fit feature scaler (essential for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, "models/ml_scaler.joblib")

        # Define 3 Classical ML Models
        models_config = {
            "Random Forest": {
                "model": RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=1),
                "use_scaled": False,
                "save_name": "models/ml_random_forest.joblib",
                "desc": "Ensemble of 120 decision trees with feature sub-sampling",
            },
            "Support Vector Machine (SVM)": {
                "model": SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42),
                "use_scaled": True,
                "save_name": "models/ml_svm.joblib",
                "desc": "Radial Basis Function (RBF) kernel margin maximizer",
            },
            "Gradient Boosting (HistGBM)": {
                "model": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42),
                "use_scaled": False,
                "save_name": "models/ml_gradient_boosting.joblib",
                "desc": "Histogram-binned gradient-boosted decision trees",
            },
        }

        comparison_results = {}
        model_names = list(models_config.keys())

        # 4. Train each model
        for m_idx, (name, cfg) in enumerate(models_config.items()):
            if progress_cb:
                progress_cb(
                    0.55 + 0.35 * (m_idx / len(model_names)),
                    f"Training {name} on video-mined features..."
                )
            clf = cfg["model"]
            x_tr = X_train_scaled if cfg["use_scaled"] else X_train
            x_te = X_test_scaled if cfg["use_scaled"] else X_test

            t_fit0 = time.perf_counter()
            clf.fit(x_tr, y_train)
            train_time = time.perf_counter() - t_fit0

            # Evaluate latency and metrics
            t_inf0 = time.perf_counter()
            y_pred = clf.predict(x_te)
            infer_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(x_te)

            acc = float(accuracy_score(y_test, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()

            joblib.dump(clf, cfg["save_name"])
            model_size_kb = float(os.path.getsize(cfg["save_name"]) / 1024.0)
            per_class_f1 = {DEFECT_CLASSES[i]: round(float(f1_c[i]), 4) for i in range(len(DEFECT_CLASSES))}

            comparison_results[name] = {
                "name": name,
                "description": cfg["desc"],
                "model_path": cfg["save_name"],
                "use_scaled": cfg["use_scaled"],
                "accuracy": round(acc, 4),
                "macro_f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "inference_ms_per_patch": round(infer_ms, 3),
                "throughput_fps": round(1000.0 / max(0.001, infer_ms), 1),
                "training_time_sec": round(train_time, 2),
                "model_size_kb": round(model_size_kb, 1),
                "per_class_f1": per_class_f1,
                "confusion_matrix": cm,
                "classes": DEFECT_CLASSES,
            }

        # 5. Persist comparison metrics JSON
        with open("models/ml_comparison_metrics.json", "w") as f:
            json.dump(comparison_results, f, indent=2)

        # 6. Build summary dataframe
        summary_rows = []
        for name, res in comparison_results.items():
            summary_rows.append({
                "Model": name,
                "Accuracy": f"{res['accuracy'] * 100:.2f}%",
                "Macro F1": f"{res['macro_f1']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall": f"{res['recall']:.4f}",
                "Latency (ms/patch)": f"{res['inference_ms_per_patch']:.3f} ms",
                "Throughput": f"{res['throughput_fps']:.0f} FPS",
                "Model Size": f"{res['model_size_kb']:.1f} KB",
                "Training Time": f"{res['training_time_sec']:.2f} s",
            })
        summary_df = pd.DataFrame(summary_rows)

        # Mined patch gallery samples (thumbnails)
        gallery_samples = []
        for rec in all_mined_records[:16]:
            gallery_samples.append({
                "class": rec["class"],
                "confidence": round(rec["confidence"], 2),
                "timestamp_s": rec["timestamp_s"],
                "patch_path": rec["patch_path"],
            })

        total_elapsed = time.perf_counter() - t0_total
        if progress_cb:
            progress_cb(1.0, f"Training complete in {total_elapsed:.1f}s!")

        # Log session to audit file
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "videos_trained_on": video_paths,
            "total_mined_patches": len(all_mined_records),
            "class_distribution": class_distribution,
            "training_duration_s": round(total_elapsed, 2),
            "results": {k: v["accuracy"] for k, v in comparison_results.items()}
        }
        with open("logs/video_training_audit.jsonl", "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        return {
            "summary_df": summary_df,
            "comparison_results": comparison_results,
            "total_mined_patches": len(all_mined_records),
            "class_distribution": class_distribution,
            "gallery_samples": gallery_samples,
            "training_duration_s": round(total_elapsed, 2),
        }


if __name__ == "__main__":
    from ml.fetch_videos import fetch_or_create_all_training_videos
    vids = fetch_or_create_all_training_videos()
    v_paths = [v["path"] for v in vids[:2]]
    print(f"[CLI] Starting VideoMLTrainer on videos: {v_paths}")
    trainer = VideoMLTrainer()
    res = trainer.train_models_from_videos(v_paths)
    print("\n[Results Summary]:")
    print(res["summary_df"].to_string(index=False))
