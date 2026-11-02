import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Retrain Classical ML Models (Random Forest, SVM, HistGBM) integrating:
1. Real WhatsApp Video 1 (Ink Stain & Light Gray Fabric)
2. Real WhatsApp Video 2 (Reddish Stain & Heather Dark Twill Fabric)
3. Industrial benchmark datasets (data/built/cls: hole, broken_thread, irregular_yarn, stain, non_defective)
"""
import json
import os
import random
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.features import CLASS_TO_ID, DEFECT_CLASSES, ID_TO_CLASS, extract_patch_features

# Set seeds
random.seed(42)
np.random.seed(42)

print("=== Starting Model Retraining with WhatsApp Videos & Benchmark Data ===")

X_list = []
y_list = []
sources = []

# 1. Load Real WhatsApp Patches
whatsapp_dir = Path("logs/mined_whatsapp_patches")
wa_stains = list(whatsapp_dir.glob("*_stain_*.jpg"))
wa_cleans = list(whatsapp_dir.glob("*_clean_*.jpg"))

print(f"Loaded from WhatsApp videos: {len(wa_stains)} real stains, {len(wa_cleans)} real clean patches")

# Augment WhatsApp stains heavily to give robust invariance
for p in wa_stains:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64))
    # Original, flips, rotations
    variants = [
        img,
        cv2.flip(img, 1),
        cv2.flip(img, 0),
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        np.clip(img.astype(np.float32) * 1.1 + 5, 0, 255).astype(np.uint8),
        np.clip(img.astype(np.float32) * 0.9 - 5, 0, 255).astype(np.uint8),
    ]
    for v in variants:
        feats = extract_patch_features(v)
        X_list.append(feats)
        y_list.append(CLASS_TO_ID["stain"])
        sources.append("whatsapp_stain")

# WhatsApp clean fabric
for p in wa_cleans[:160]:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64))
    variants = [img, cv2.flip(img, 1), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)]
    for v in variants:
        feats = extract_patch_features(v)
        X_list.append(feats)
        y_list.append(CLASS_TO_ID["non_defective"])
        sources.append("whatsapp_clean")

# 2. Load Benchmark Dataset Patches from data/built/cls/train and val
cls_root = Path("data/built/cls")
max_per_class = 200

for cls_name in DEFECT_CLASSES:
    cid = CLASS_TO_ID[cls_name]
    files = []
    for split in ["train", "val"]:
        s_dir = cls_root / split / cls_name
        if s_dir.exists():
            files.extend(list(s_dir.glob("*.jpg")))
    random.shuffle(files)
    selected = files[:max_per_class]
    print(f"Loaded {len(selected)} patches from benchmark for '{cls_name}'")
    for f in selected:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        feats = extract_patch_features(img)
        X_list.append(feats)
        y_list.append(cid)
        sources.append(f"benchmark_{cls_name}")
        # Add 1 flip
        feats_fl = extract_patch_features(cv2.flip(img, 1))
        X_list.append(feats_fl)
        y_list.append(cid)
        sources.append(f"benchmark_{cls_name}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

print(f"\nTotal Dataset Size: {len(X)} feature vectors across {len(DEFECT_CLASSES)} classes.")
for cname, cid in CLASS_TO_ID.items():
    print(f"  Class '{cname}' (id={cid}): {np.sum(y == cid)} samples")

# 3. Train / Validation Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 4. Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/ml_scaler.joblib")
print("Saved models/ml_scaler.joblib")

# 5. Train Models
models_config = {
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_split=3, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "use_scaled": False,
        "path": "models/ml_random_forest.joblib"
    },
    "Support Vector Machine (SVM)": {
        "model": SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
        "use_scaled": True,
        "path": "models/ml_svm.joblib"
    },
    "Gradient Boosting (HistGBM)": {
        "model": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_depth=12, class_weight="balanced", random_state=42
        ),
        "use_scaled": False,
        "path": "models/ml_gradient_boosting.joblib"
    }
}

metrics_summary = {}

for name, cfg in models_config.items():
    t0 = time.time()
    clf = cfg["model"]
    X_tr = X_train_scaled if cfg["use_scaled"] else X_train
    X_te = X_test_scaled if cfg["use_scaled"] else X_test
    
    print(f"\n--- Training {name} ---")
    clf.fit(X_tr, y_train)
    t_train = round(time.time() - t0, 2)
    
    # Evaluate
    preds = clf.predict(X_te)
    report = classification_report(y_test, preds, target_names=DEFECT_CLASSES, output_dict=True)
    acc = float(report["accuracy"])
    print(f"Accuracy: {acc:.4f} (trained in {t_train}s)")
    for cname in DEFECT_CLASSES:
        f1 = report[cname]["f1-score"]
        print(f"  {cname:15s}: F1={f1:.3f}, Precision={report[cname]['precision']:.3f}, Recall={report[cname]['recall']:.3f}")
        
    joblib.dump(clf, cfg["path"])
    print(f"Saved {cfg['path']}")
    
    metrics_summary[name] = {
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "training_time_s": t_train,
        "per_class": {
            cname: {
                "precision": float(report[cname]["precision"]),
                "recall": float(report[cname]["recall"]),
                "f1": float(report[cname]["f1-score"])
            }
            for cname in DEFECT_CLASSES
        }
    }

# Save metrics JSON
with open("models/ml_comparison_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved models/ml_comparison_metrics.json")
print("=== Training Complete Successfully! ===")
