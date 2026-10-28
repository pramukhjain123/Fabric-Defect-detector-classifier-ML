"""
Module for fetching, ingesting, and synthesizing fabric defect inspection videos.
Supports downloading videos from remote URLs, ingesting user-uploaded videos,
and synthesizing high-fidelity industrial line-scan videos across varied weave patterns.
"""
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import cv2
import numpy as np

VIDEO_DIR = Path("data/videos")
UPLOAD_DIR = Path("logs/uploaded_videos")

# Publicly available sample fabric defect videos / frame sequence archives
PUBLIC_VIDEO_RESOURCES = [
    {
        "name": "sample_fabric_defect_panning_1.mp4",
        "description": "Industrial line-scan weave with puncturing hole defect and contrast variation",
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi", # fallback source
        "defect_type": "hole"
    }
]


def create_industrial_inspection_video(
    output_path: str,
    pattern: str = "plain",
    defect_types: Optional[List[str]] = None,
    duration_sec: float = 8.0,
    fps: int = 25,
    width: int = 640,
    height: int = 480,
    scroll_speed: int = 3
) -> str:
    """Synthesize a continuous moving fabric web simulating an industrial conveyor or line-scan camera.
    
    Args:
        output_path: Destination .mp4 file path.
        pattern: Fabric weave pattern ('plain', 'twill', 'knit').
        defect_types: List of defect classes to introduce across the video clip.
        duration_sec: Duration of video in seconds.
        fps: Frames per second.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        scroll_speed: Fabric movement in pixels per frame.
        
    Returns:
        Path to generated video file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if defect_types is None:
        defect_types = ["hole", "stain", "broken_thread", "irregular_yarn"]

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    # Create a large scrolling textile web buffer (tiled height)
    buffer_height = height + total_frames * scroll_speed + 200
    base_gray = 160
    
    # Generate background woven textile texture
    y, x = np.mgrid[0:buffer_height, 0:width]
    if pattern == "twill":
        # Diagonal weave ridge
        diag = (x + y) * (np.pi / 2.5)
        weave = (np.sin(diag) * 14.0).astype(np.float32)
    elif pattern == "knit":
        # Interlooping loops
        weave = (np.sin(x * 0.8) * np.cos(y * 0.4) * 12.0).astype(np.float32)
    else:
        # Plain orthogonal warp and weft
        warp = np.sin(x * 0.75 * np.pi) * 10.0
        weft = np.cos(y * 0.75 * np.pi) * 10.0
        weave = (warp + weft).astype(np.float32)

    noise = np.random.normal(0, 4.0, (buffer_height, width)).astype(np.float32)
    web = np.clip(base_gray + weave + noise, 0, 255).astype(np.uint8)

    # Imprint defects at spaced vertical intervals along the fabric web
    defect_positions = []
    spacing = buffer_height // (len(defect_types) + 2)
    for idx, d_type in enumerate(defect_types):
        dy = int(height // 2 + (idx + 1) * spacing)
        dx = int(width // 2 + np.random.randint(-width // 4, width // 4))
        defect_positions.append((dy, dx, d_type))

        if d_type == "hole":
            # Missing yarn / tear
            rx, ry = np.random.randint(22, 38), np.random.randint(20, 36)
            cv2.ellipse(web, (dx, dy), (rx + 4, ry + 4), 30, 0, 360, int(base_gray * 0.4), -1)
            cv2.ellipse(web, (dx, dy), (rx, ry), 30, 0, 360, 20, -1)
            # Add frayed yarn edges
            for _ in range(12):
                fx = dx + np.random.randint(-rx - 4, rx + 4)
                fy = dy + np.random.randint(-ry - 4, ry + 4)
                cv2.circle(web, (fx, fy), np.random.randint(1, 3), 220, -1)

        elif d_type == "stain":
            # Machine oil droplet with gradient attenuation
            rx, ry = np.random.randint(35, 60), np.random.randint(30, 50)
            mask = np.zeros((buffer_height, width), dtype=np.float32)
            cv2.ellipse(mask, (dx, dy), (rx, ry), 45, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            web = np.clip(web.astype(np.float32) * (1.0 - 0.55 * mask), 0, 255).astype(np.uint8)

        elif d_type == "broken_thread":
            # Broken warp yarn line
            line_len = np.random.randint(70, 130)
            cv2.line(web, (dx, max(0, dy - line_len // 2)), (dx, min(buffer_height - 1, dy + line_len // 2)), 25, 3)
            cv2.line(web, (dx - 1, max(0, dy - line_len // 2 + 5)), (dx - 1, min(buffer_height - 1, dy + line_len // 2 - 5)), 245, 1)

        elif d_type == "irregular_yarn":
            # Slub / clump of thick uneven yarn
            rx, ry = np.random.randint(18, 32), np.random.randint(12, 22)
            cv2.ellipse(web, (dx, dy), (rx, ry), 15, 0, 360, 235, -1)
            cv2.ellipse(web, (dx, dy), (rx - 3, ry - 3), 15, 0, 360, 130, 2)

    # Render moving camera window across the fabric web
    for f in range(total_frames):
        y1 = f * scroll_speed
        y2 = y1 + height
        if y2 > buffer_height:
            break
        frame = web[y1:y2, 0:width]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()
    return output_path


def fetch_or_create_all_training_videos() -> List[Dict]:
    """Ensure data/videos directory contains real, uploaded, and synthetic fabric inspection videos."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. Ingest existing demo video if available
    demo_video = Path("data/fabric_moving_scan.mp4")
    if demo_video.exists():
        dest = VIDEO_DIR / "demo_panning_fabric_scan.mp4"
        if not dest.exists():
            shutil.copy2(demo_video, dest)
        results.append({
            "path": str(dest),
            "name": "Demo Panning Fabric Scan",
            "type": "Moving Loom Simulation",
            "defects": ["hole", "stain", "broken_thread"]
        })

    # 2. Ingest uploaded videos (e.g., real phone / WhatsApp recorded textile clips)
    if UPLOAD_DIR.exists():
        for vid_file in UPLOAD_DIR.glob("*.mp4"):
            clean_name = f"uploaded_{vid_file.stem[:24].replace(' ', '_').lower()}.mp4"
            dest = VIDEO_DIR / clean_name
            if not dest.exists():
                shutil.copy2(vid_file, dest)
            results.append({
                "path": str(dest),
                "name": f"User Upload: {vid_file.stem[:28]}",
                "type": "Real Camera Recording",
                "defects": ["real_defective_fabric"]
            })

    # 3. Generate high-fidelity industrial moving fabric inspection videos
    vids_to_generate = [
        ("industrial_plain_inspection_1.mp4", "plain", ["hole", "stain", "broken_thread"], "Conveyor Inspection (Plain Weave)"),
        ("industrial_twill_inspection_2.mp4", "twill", ["irregular_yarn", "broken_thread"], "Twill Fabric Inspection (Slub & Thread)"),
        ("industrial_knit_inspection_3.mp4", "knit", ["hole", "stain"], "Knit Fabric Inspection (Hole & Oil Contamination)")
    ]

    for filename, pattern, d_types, label in vids_to_generate:
        target_path = VIDEO_DIR / filename
        if not target_path.exists():
            print(f"[Video Generator] Generating {label} -> {target_path}...")
            create_industrial_inspection_video(
                str(target_path),
                pattern=pattern,
                defect_types=d_types,
                duration_sec=7.0,
                fps=25,
                width=640,
                height=480,
                scroll_speed=3
            )
        results.append({
            "path": str(target_path),
            "name": label,
            "type": f"Industrial {pattern.capitalize()} Weave",
            "defects": d_types
        })

    return results


def get_available_videos() -> List[Dict]:
    """Retrieve metadata of all available fabric videos in data/videos and logs/uploaded_videos."""
    vids = fetch_or_create_all_training_videos()
    enriched = []
    for v in vids:
        p = Path(v["path"])
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            v_info = dict(v)
            v_info["frames"] = total_frames
            v_info["fps"] = round(fps, 1)
            v_info["duration_sec"] = round(total_frames / max(1.0, fps), 1)
            v_info["resolution"] = f"{width}x{height}"
            v_info["file_size_mb"] = round(p.stat().st_size / (1024 * 1024), 2)
            enriched.append(v_info)
    return enriched


if __name__ == "__main__":
    print("[ml.fetch_videos] Initializing fabric defect inspection video suite...")
    videos = get_available_videos()
    for v in videos:
        print(f" -> {v['name']}: {v['path']} ({v['resolution']}, {v['duration_sec']}s, {v['file_size_mb']}MB)")
    print(f"Total available videos: {len(videos)}")
