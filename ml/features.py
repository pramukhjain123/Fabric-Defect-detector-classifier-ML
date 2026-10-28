"""
Handcrafted feature extraction for Classical Machine Learning fabric defect inspection.
Extracts GLCM texture descriptors, LBP histograms, statistical intensity moments,
Sobel edge gradients, and morphological properties.
"""
from typing import Dict, List, Tuple
import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Pre-defined classes
DEFECT_CLASSES = ["broken_thread", "hole", "irregular_yarn", "non_defective", "stain"]
CLASS_TO_ID = {name: i for i, name in enumerate(DEFECT_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(DEFECT_CLASSES)}


def extract_patch_features(patch: np.ndarray) -> np.ndarray:
    """Extract a comprehensive 64-dimensional handcrafted feature vector from an image patch.
    
    Args:
        patch: 2D grayscale or 3D BGR image (uint8).
        
    Returns:
        1D float32 numpy array of length 64.
    """
    if patch is None or patch.size == 0:
        return np.zeros(64, dtype=np.float32)

    # Standardize to 64x64 grayscale uint8
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch

    if gray.shape != (64, 64):
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    feats = []

    # ---------------- 1. Statistical Moments (8 features) ----------------
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray)) + 1e-6
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    iqr = float(p90 - p10)
    skewness = float(stats.skew(gray.ravel()))
    kurt = float(stats.kurtosis(gray.ravel()))

    feats.extend([mean_val, std_val, float(p10), float(p50), float(p90), iqr, skewness, kurt])

    # ---------------- 2. GLCM Texture Descriptors (12 features) ----------
    # Quantize to 16 gray levels for fast, invariant co-occurrence calculation
    gray_q = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop)
        feats.append(float(np.mean(vals)))
        feats.append(float(np.std(vals)))

    # ---------------- 3. Local Binary Patterns (LBP) (28 features) -------
    # P=8, R=1 uniform LBP (10 bins)
    lbp8 = local_binary_pattern(gray, P=8, R=1.0, method="uniform")
    hist8, _ = np.histogram(lbp8.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(hist8.tolist())

    # P=16, R=2 uniform LBP (18 bins)
    lbp16 = local_binary_pattern(gray, P=16, R=2.0, method="uniform")
    hist16, _ = np.histogram(lbp16.ravel(), bins=18, range=(0, 18), density=True)
    feats.extend(hist16.tolist())

    # ---------------- 4. Gradient / Edge Descriptors (11 features) -------
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    feats.append(float(np.mean(mag)))
    feats.append(float(np.std(mag)))
    feats.append(float(np.sum(mag ** 2) / (64 * 64 * 255.0)))

    # 8-bin gradient orientation histogram weighted by magnitude
    ang_bins = np.linspace(0, 360, 9)
    hist_ang, _ = np.histogram(ang.ravel(), bins=ang_bins, weights=mag.ravel(), density=True)
    feats.extend(hist_ang.tolist())

    # ---------------- 5. Morphological Defect Metrics (5 features) -------
    # Difference from background mean
    diff = cv2.absdiff(gray, int(mean_val))
    _, thresh = cv2.threshold(diff, int(std_val * 1.5 + 10), 255, cv2.THRESH_BINARY)
    defect_ratio = float(np.count_nonzero(thresh) / (64 * 64))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_components = float(len(contours))
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c_max))
        perimeter = float(cv2.arcLength(c_max, True)) + 1e-5
        compactness = float(4 * np.pi * area / (perimeter ** 2))
        _, _, w_b, h_b = cv2.boundingRect(c_max)
        aspect_ratio = float(w_b / max(1, h_b))
    else:
        area, compactness, aspect_ratio = 0.0, 0.0, 1.0

    feats.extend([defect_ratio, num_components, area, compactness, aspect_ratio])

    arr = np.array(feats, dtype=np.float32)
    # Sanitize any accidental nan or inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return arr


def get_feature_names() -> List[str]:
    """Return explicit names for the 64 handcrafted features."""
    names = [
        "stat_mean", "stat_std", "stat_p10", "stat_p50", "stat_p90", "stat_iqr", "stat_skew", "stat_kurt",
    ]
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        names.append(f"glcm_{prop}_mean")
        names.append(f"glcm_{prop}_std")
    for i in range(10):
        names.append(f"lbp8_bin_{i}")
    for i in range(18):
        names.append(f"lbp16_bin_{i}")
    names.extend(["grad_mag_mean", "grad_mag_std", "grad_energy"])
    for i in range(8):
        names.append(f"grad_ang_bin_{i}")
    names.extend(["morph_defect_ratio", "morph_num_components", "morph_max_area", "morph_compactness", "morph_aspect_ratio"])
    return names
