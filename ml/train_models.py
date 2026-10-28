"""
Dataset generator and trainer for 3 Classical Machine Learning models:
1. Random Forest (RF)
2. Support Vector Machine (SVM)
3. Histogram-based Gradient Boosting (HistGBM)

Generates balanced fabric defect datasets, extracts 64 handcrafted features,
trains each model, evaluates on a held-out test split, and outputs full comparative metrics.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.features import extract_patch_features, DEFECT_CLASSES, CLASS_TO_ID, ID_TO_CLASS, get_feature_names


# ---------------- Synthetic Fabric Sample Synthesizer -------------------------

def create_synthetic_fabric_patch(cls_name: str, size: int = 64, seed: int = None) -> np.ndarray:
    """Generate a high-fidelity synthetic fabric patch with plain/twill weave and defect."""
    rng = np.random.default_rng(seed)
    base_gray = rng.integers(140, 190)

    # 1. Base woven structure (Warp & Weft frequency grid)
    y, x = np.mgrid[0:size, 0:size]
    freq = rng.uniform(0.5, 0.9)
    warp = np.sin(x * freq * np.pi) * rng.uniform(8, 18)
    weft = np.cos(y * freq * np.pi) * rng.uniform(8, 18)
    noise = rng.normal(0, rng.uniform(3, 7), (size, size))
    patch = np.clip(base_gray + warp + weft + noise, 0, 255).astype(np.uint8)

    cx = int(size // 2 + rng.integers(-8, 9))
    cy = int(size // 2 + rng.integers(-8, 9))

    if cls_name == "hole":
        # Missing material / puncture or tear:
        hole_type = rng.choice(["puncture_dark", "puncture_backlit", "torn_slit"])
        rx = int(rng.integers(size // 8, size // 3))
        ry = int(rng.integers(size // 8, size // 3))
        angle = int(rng.integers(0, 180))

        if hole_type == "puncture_dark":
            cv2.ellipse(patch, (cx, cy), (rx + 3, ry + 3), angle, 0, 360, int(base_gray * 0.4), -1)
            cv2.ellipse(patch, (cx, cy), (rx, ry), angle, 0, 360, int(rng.integers(15, 60)), -1)
        elif hole_type == "puncture_backlit":
            # Hole with light showing through (backlit/white table) + dark/blue perimeter yarn distortion
            cv2.ellipse(patch, (cx, cy), (rx + 4, ry + 4), angle, 0, 360, int(rng.integers(30, 80)), 2)
            cv2.ellipse(patch, (cx, cy), (rx, ry), angle, 0, 360, int(rng.integers(235, 255)), -1)
            cv2.ellipse(patch, (cx, cy), (rx + 2, ry + 2), angle, 0, 360, int(rng.integers(40, 100)), 1)
        else: # torn_slit
            slit_w = int(rng.integers(size // 3, size - 10))
            slit_h = int(rng.integers(4, 14))
            cv2.ellipse(patch, (cx, cy), (slit_w // 2, slit_h // 2), int(rng.integers(-15, 15)), 0, 360, int(rng.integers(30, 75)), -1)
            # fray threads across the slit
            for _ in range(rng.integers(2, 5)):
                tx = int(cx + rng.integers(-slit_w // 3, slit_w // 3))
                cv2.line(patch, (tx, cy - slit_h // 2), (tx + rng.integers(-4, 5), cy + slit_h // 2), int(rng.integers(180, 240)), 1)


    elif cls_name == "stain":
        # Smooth photometric contaminant (oil/grease drop)
        rx = int(rng.integers(size // 6, size // 3))
        ry = int(rng.integers(size // 6, size // 3))
        stain_mask = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(stain_mask, (cx, cy), (rx, ry), int(rng.integers(0, 180)), 0, 360, 1.0, -1)
        stain_mask = cv2.GaussianBlur(stain_mask, (15, 15), 0)
        stain_intensity = float(rng.uniform(0.35, 0.75))
        patch = np.clip(patch.astype(np.float32) * (1.0 - (1.0 - stain_intensity) * stain_mask), 0, 255).astype(np.uint8)

    elif cls_name == "broken_thread":
        # Sharp horizontal or vertical missing/pulled thread
        is_horiz = bool(rng.choice([True, False]))
        if is_horiz:
            y_line = int(cy)
            cv2.line(patch, (max(0, cx - 24), y_line), (min(size - 1, cx + 24), y_line), int(rng.integers(20, 50)), 2)
            cv2.line(patch, (max(0, cx - 18), y_line - 1), (min(size - 1, cx + 18), y_line - 1), int(rng.integers(210, 250)), 1)
        else:
            x_line = int(cx)
            cv2.line(patch, (x_line, max(0, cy - 24)), (x_line, min(size - 1, cy + 24)), int(rng.integers(20, 50)), 2)
            cv2.line(patch, (x_line - 1, max(0, cy - 18)), (x_line - 1, min(size - 1, cy + 18)), int(rng.integers(210, 250)), 1)

    elif cls_name == "irregular_yarn":
        # Local bunching / yarn knot (slub defect)
        rx = int(rng.integers(size // 10, size // 5))
        ry = int(rng.integers(size // 10, size // 5))
        cv2.ellipse(patch, (cx, cy), (rx + 4, ry + 4), int(rng.integers(0, 180)), 0, 360, int(rng.integers(200, 245)), -1)
        cv2.ellipse(patch, (cx, cy), (rx, ry), int(rng.integers(0, 180)), 0, 360, int(rng.integers(120, 160)), 2)

    # non_defective stays purely the woven background
    return patch


def augment_crop_variants(crop: np.ndarray, class_name: str, target_size: int = 64) -> List[Tuple[np.ndarray, int]]:
    """Generate rotation, flip, scaling, and contrast variants of a real fabric crop."""
    cid = CLASS_TO_ID[class_name]
    variants = []
    if crop is None or crop.size == 0:
        return variants
    
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
    for angle in [0, 90, 180, 270]:
        if angle == 90:
            rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot = cv2.rotate(crop, cv2.ROTATE_180)
        elif angle == 270:
            rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rot = crop
            
        for flip in [False, True]:
            f = cv2.flip(rot, 1) if flip else rot
            for scale in [0.85, 1.0, 1.15]:
                h, w = f.shape[:2]
                new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
                res = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                patch64 = cv2.resize(res, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
                # Base variant
                variants.append((extract_patch_features(patch64), cid))
                
                # Small brightness perturbation
                for alpha, beta in [(0.92, -10), (1.08, 10)]:
                    perturbed = np.clip(patch64.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                    variants.append((extract_patch_features(perturbed), cid))
                    
    return variants


def generate_dataset(samples_per_class: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced feature matrix X and label vector y blending synthetic and real fabric samples."""
    print(f"[Dataset] Generating balanced dataset across {len(DEFECT_CLASSES)} classes...")
    X_list = []
    y_list = []

    # 1. Base synthetic samples
    for class_id, class_name in enumerate(DEFECT_CLASSES):
        for i in range(samples_per_class):
            patch = create_synthetic_fabric_patch(class_name, size=64, seed=class_id * 1000 + i)
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            patch = np.clip(patch * alpha + beta, 0, 255).astype(np.uint8)
            feat = extract_patch_features(patch)
            X_list.append(feat)
            y_list.append(class_id)

    # 2. Project root reference patches
    ref_patches = [
        ("patch_hole.jpg", "hole"),
        ("patch_stain.jpg", "stain"),
        ("patch_clean.jpg", "non_defective"),
        ("patch_knot.jpg", "irregular_yarn")
    ]
    for p_name, p_cls in ref_patches:
        p_path = Path(p_name)
        if p_path.exists():
            im = cv2.imread(str(p_path), cv2.IMREAD_GRAYSCALE)
            for feat, cid in augment_crop_variants(im, p_cls):
                X_list.append(feat)
                y_list.append(cid)

    # 3. test_user_fabric.jpg crops
    real_user_img = Path("test_user_fabric.jpg")
    if real_user_img.exists():
        im = cv2.imread(str(real_user_img), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            user_crops = [
                (im[240:304, 70:134], "hole"),
                (im[620:684, 460:524], "stain"),
                (im[600:664, 230:294], "irregular_yarn"),
                (im[100:164, 650:714], "non_defective"),
            ]
            for crop, c_name in user_crops:
                for feat, cid in augment_crop_variants(crop, c_name):
                    X_list.append(feat)
                    y_list.append(cid)

    # 4. Verified real video defect crops
    real_holes_dir = Path("logs/real_holes")
    if real_holes_dir.exists():
        for hp in real_holes_dir.glob("*.jpg"):
            im = cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                for feat, cid in augment_crop_variants(im, "hole"):
                    X_list.append(feat)
                    y_list.append(cid)

    # 4b. Verified real video stain crops
    real_stains_dir = Path("logs/real_stains")
    if real_stains_dir.exists():
        for sp in real_stains_dir.glob("*.jpg"):
            im = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                for feat, cid in augment_crop_variants(im, "stain"):
                    X_list.append(feat)
                    y_list.append(cid)

    # 5. Clean fabric patches sampled from video frames
    sample_frames = list(Path("logs/video_inspect_frames").glob("frame_*.jpg"))
    for sf in sample_frames[:4]:
        frame_im = cv2.imread(str(sf), cv2.IMREAD_GRAYSCALE)
        if frame_im is not None:
            for yc in [120, 220, 520]:
                for xc in [120, 220, 320]:
                    clean_patch = frame_im[yc:yc+64, xc:xc+64]
                    if clean_patch.shape == (64, 64):
                        for feat, cid in augment_crop_variants(clean_patch, "non_defective")[:12]:
                            X_list.append(feat)
                            y_list.append(cid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[Dataset] Total dataset built: Shape={X.shape}, Class distribution={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def train_and_evaluate_models():
    """Train Random Forest, SVM, and Gradient Boosting; evaluate and persist."""
    Path("models").mkdir(parents=True, exist_ok=True)
    X, y = generate_dataset(samples_per_class=200)

    # Train / Test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit feature scaler (essential for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/ml_scaler.joblib")

    models_config = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=1),
            "use_scaled": False,
            "save_name": "models/ml_random_forest.joblib",
            "desc": "Ensemble of 120 decision trees with feature sub-sampling"
        },
        "Support Vector Machine (SVM)": {
            "model": SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42),
            "use_scaled": True,
            "save_name": "models/ml_svm.joblib",
            "desc": "Radial Basis Function (RBF) kernel margin maximizer"
        },
        "Gradient Boosting (HistGBM)": {
            "model": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42),
            "use_scaled": False,
            "save_name": "models/ml_gradient_boosting.joblib",
            "desc": "Histogram-binned gradient-boosted decision trees"
        }
    }

    comparison_results = {}

    print("\n" + "=" * 70)
    print("      TRAINING & BENCHMARKING 3 CLASSICAL ML FABRIC DEFECT MODELS")
    print("=" * 70)

    for name, cfg in models_config.items():
        print(f"\n[Training] {name} ({cfg['desc']})...")
        clf = cfg["model"]
        x_tr = X_train_scaled if cfg["use_scaled"] else X_train
        x_te = X_test_scaled if cfg["use_scaled"] else X_test

        t0 = time.perf_counter()
        clf.fit(x_tr, y_train)
        train_time = time.perf_counter() - t0

        # Benchmark inference latency
        t1 = time.perf_counter()
        y_pred = clf.predict(x_te)
        infer_total_ms = (time.perf_counter() - t1) * 1000.0
        ms_per_patch = infer_total_ms / len(x_te)

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Save model
        joblib.dump(clf, cfg["save_name"])
        model_size_kb = float(os.path.getsize(cfg["save_name"]) / 1024.0)

        per_class_f1 = {DEFECT_CLASSES[i]: round(float(f1_c[i]), 4) for i in range(len(DEFECT_CLASSES))}

        print(f"  -> Accuracy:     {acc * 100:.2f}%")
        print(f"  -> Macro F1:     {f1:.4f}")
        print(f"  -> Precision:    {prec:.4f}")
        print(f"  -> Recall:       {rec:.4f}")
        print(f"  -> Latency:      {ms_per_patch:.3f} ms / patch ({1000.0 / ms_per_patch:.0f} FPS)")
        print(f"  -> Model Size:   {model_size_kb:.1f} KB")

        comparison_results[name] = {
            "name": name,
            "description": cfg["desc"],
            "model_path": cfg["save_name"],
            "use_scaled": cfg["use_scaled"],
            "accuracy": round(acc, 4),
            "macro_f1": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "inference_ms_per_patch": round(ms_per_patch, 3),
            "throughput_fps": round(1000.0 / ms_per_patch, 1),
            "training_time_sec": round(train_time, 2),
            "model_size_kb": round(model_size_kb, 1),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "classes": DEFECT_CLASSES
        }

    # Save comparison report JSON
    with open("models/ml_comparison_metrics.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARATIVE STUDY REPORT SAVED to models/ml_comparison_metrics.json")
    print("ALL 3 MODELS SUCCESSFULLY TRAINED AND SERIALIZED!")
    print("=" * 70)
    return comparison_results


if __name__ == "__main__":
    train_and_evaluate_models()
