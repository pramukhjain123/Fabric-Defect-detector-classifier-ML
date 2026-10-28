"""
Generate a realistic fabric video panning left-to-right (line-scan style) with defects.
Useful for offline testing and demonstration of the video inspection pipeline.
"""
from pathlib import Path
import cv2
import numpy as np


def generate_panning_fabric_video(output_path: str = "data/fabric_moving_scan.mp4", duration_sec: int = 5, fps: int = 25):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame_w, frame_h = 640, 480
    total_frames = duration_sec * fps

    # Create a long fabric strip that will move horizontally
    strip_w = frame_w + total_frames * 6
    strip_h = frame_h
    strip = np.full((strip_h, strip_w, 3), 160, dtype=np.uint8)

    # Add weave texture
    y, x = np.mgrid[0:strip_h, 0:strip_w]
    warp = (np.sin(x * 0.7 * np.pi) * 12).astype(np.int16)
    weft = (np.cos(y * 0.7 * np.pi) * 12).astype(np.int16)
    texture = np.clip(160 + warp + weft, 0, 255).astype(np.uint8)
    for c in range(3):
        strip[:, :, c] = texture

    # Embed Defect 1: A hole (dark elliptical void) around x=400, y=240
    cv2.ellipse(strip, (400, 240), (22, 22), 0, 0, 360, (25, 25, 25), -1)

    # Embed Defect 2: An oil stain around x=850, y=300
    stain_overlay = np.zeros((strip_h, strip_w), dtype=np.float32)
    cv2.ellipse(stain_overlay, (850, 300), (38, 25), 30, 0, 360, 1.0, -1)
    stain_overlay = cv2.GaussianBlur(stain_overlay, (21, 21), 0)
    for c in range(3):
        strip[:, :, c] = np.clip(strip[:, :, c].astype(np.float32) * (1.0 - 0.55 * stain_overlay), 0, 255).astype(np.uint8)

    # Embed Defect 3: A broken thread around x=1300, y=180
    cv2.line(strip, (1300, 120), (1300, 240), (40, 40, 40), 2)
    cv2.line(strip, (1301, 120), (1301, 240), (240, 240, 240), 1)

    # Write video by panning a 640x480 window across the strip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for f in range(total_frames):
        offset_x = f * 6
        crop = np.ascontiguousarray(strip[0:frame_h, offset_x:offset_x + frame_w])
        writer.write(crop)

    writer.release()
    print(f"[Demo Video] Generated {total_frames} frames moving fabric video: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_panning_fabric_video()
