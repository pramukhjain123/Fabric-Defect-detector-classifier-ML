"""
Quick video validator script.
Runs comparative analysis across Random Forest, SVM, and Gradient Boosting (HistGBM).
"""
import sys
import os
from pathlib import Path
import json

from ml.video_inspector import run_comparative_video_study

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "logs/uploaded_videos/WhatsApp Video 2026-10-28 at 20.32.15.mp4"
    conf_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40
    
    print("==================================================")
    print(f"VALIDATING VIDEO: {video_path}")
    print(f"CONFIDENCE THRESHOLD: {conf_thresh}")
    print("==================================================")
    
    def on_prog(pct, msg):
        print(f"[{int(pct*100):3d}%] {msg}")

    results = run_comparative_video_study(
        video_path=video_path,
        conf_threshold=conf_thresh,
        progress_cb=on_prog
    )
    
    df = results["comparative_df"]
    print("\n" + "="*50)
    print("COMPARATIVE STUDY RESULTS (ALL 3 ML MODELS)")
    print("="*50)
    print(df.to_string(index=False))
    
    # Save validation report
    report_path = Path("logs/video_validation_report.json")
    summary_data = {
        "video": str(video_path),
        "confidence_threshold": conf_thresh,
        "models": {}
    }
    for m_name, m_res in results["model_results"].items():
        summary_data["models"][m_name] = {
            "stats": m_res["stats"],
            "events_count": len(m_res["events"]),
            "events": m_res["events"][:10], # sample top 10
            "output_video": m_res["output_video_path"]
        }
    with open(report_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nDetailed validation audit saved to: {report_path}")

if __name__ == "__main__":
    main()
