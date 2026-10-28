"""
Command-Line Interface (CLI) to train Classical Machine Learning models directly from fabric videos.
Can ingest videos from internet/repository, mine candidate and confirmed defect patches,
extract 64 handcrafted features, train Random Forest, SVM, and HistGBM models,
and evaluate performance.

Usage:
    python train_from_video.py                              # Auto-train on all available videos
    python train_from_video.py --video path/to/video.mp4    # Train on specific video
    python train_from_video.py --fetch                      # Fetch/generate latest online inspection videos
"""
import argparse
import sys
from pathlib import Path

from ml.fetch_videos import fetch_or_create_all_training_videos, get_available_videos
from ml.video_trainer import VideoMLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Classical ML Fabric Defect Classifier directly from videos.")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to specific fabric video (.mp4, .avi).")
    parser.add_argument("--fetch", action="store_true", help="Fetch or generate latest online inspection video feeds.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for mining defect patches.")
    args = parser.parse_args()

    print("=" * 75)
    print("      FABRIC DEFECT VIDEO SELF-LEARNING & CLASSICAL ML TRAINER")
    print("=" * 75)

    if args.fetch:
        print("[Fetch] Discovering and synthesizing online inspection videos...")
        vids = fetch_or_create_all_training_videos()
        print(f"[Fetch] {len(vids)} videos ready in data/videos/")

    if args.video:
        v_path = Path(args.video)
        if not v_path.exists():
            print(f"[Error] Specified video does not exist: {args.video}")
            sys.exit(1)
        video_paths = [str(v_path)]
    else:
        vids = get_available_videos()
        if not vids:
            print("[Info] No videos found in data/videos/. Initializing default video suite...")
            vids = fetch_or_create_all_training_videos()
        video_paths = [v["path"] for v in vids]

    print(f"\n[Training Input] {len(video_paths)} video(s) queued for patch mining & ML training:")
    for vp in video_paths:
        print(f"  * {Path(vp).name}")

    print("\n[Pipeline] Initializing VideoMLTrainer (Extracting GLCM, LBP, Moments, Gradients)...")
    trainer = VideoMLTrainer(detector_conf=args.conf)

    def print_progress(pct, msg):
        bar_len = 30
        filled = int(round(pct * bar_len))
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {int(pct*100):3d}% | {msg[:45]:<45}")
        sys.stdout.flush()

    results = trainer.train_models_from_videos(video_paths, progress_cb=print_progress)
    print("\n\n" + "=" * 75)
    print("                      COMPARATIVE STUDY RESULTS")
    print("=" * 75)
    print(results["summary_df"].to_string(index=False))
    print("=" * 75)
    print(f"Total Video Patches Mined: {results['total_mined_patches']}")
    print(f"Class Distribution: {results['class_distribution']}")
    print(f"Total Training Time: {results['training_duration_s']}s")
    print("\nTrained models saved to: models/ml_random_forest.joblib, models/ml_svm.joblib, models/ml_gradient_boosting.joblib")
    print("Metrics report saved to: models/ml_comparison_metrics.json")
    print("Audit log appended to:   logs/video_training_audit.jsonl\n")


if __name__ == "__main__":
    main()
