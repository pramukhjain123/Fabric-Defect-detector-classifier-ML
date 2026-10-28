"""Unit and integration tests for Classical Machine Learning pipeline."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features import extract_patch_features, get_feature_names, DEFECT_CLASSES
from ml.video_inspector import ClassicalMLVideoInspector, run_comparative_video_study


def test_feature_extraction():
    patch = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    feats = extract_patch_features(patch)
    names = get_feature_names()
    assert feats.shape == (64,)
    assert len(names) == 64
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_models_exist_and_predict():
    patch = np.full((64, 64), 160, dtype=np.uint8)
    for model_name in ["Random Forest", "Support Vector Machine (SVM)", "Gradient Boosting (HistGBM)"]:
        inspector = ClassicalMLVideoInspector(model_name=model_name)
        cls_name, conf = inspector.predict_patch(patch)
        assert cls_name in DEFECT_CLASSES
        assert 0.0 <= conf <= 1.0


def test_video_inspector_pipeline():
    demo_video = Path("data/fabric_moving_scan.mp4")
    if not demo_video.exists():
        from ml.create_demo_video import generate_panning_fabric_video
        generate_panning_fabric_video()

    inspector = ClassicalMLVideoInspector(model_name="Random Forest")
    res = inspector.inspect_video(str(demo_video), conf_threshold=0.35)
    assert res is not None
    assert "stats" in res
    assert res["stats"]["total_frames"] > 0
    assert "events_df" in res


def test_comparative_study():
    demo_video = Path("data/fabric_moving_scan.mp4")
    if not demo_video.exists():
        from ml.create_demo_video import generate_panning_fabric_video
        generate_panning_fabric_video()

    study = run_comparative_video_study(str(demo_video), conf_threshold=0.35)
    assert "comparative_df" in study
    assert len(study["comparative_df"]) == 3
    assert "ML Model" in study["comparative_df"].columns
