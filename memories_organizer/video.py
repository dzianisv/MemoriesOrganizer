import cv2
from pathlib import Path
from typing import List


def extract_frames(video_path: str, output_dir: str, every_seconds: int = 30) -> List[str]:
    """Extract frames every `every_seconds` from video to `output_dir`.

    Returns list of frame file paths.
    """
    path = Path(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * every_seconds)
    frame_count = 0
    saved_paths = []

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_file = output / f"frame_{index:04d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_paths.append(str(frame_file))
            index += 1
        frame_count += 1

    cap.release()
    return saved_paths
