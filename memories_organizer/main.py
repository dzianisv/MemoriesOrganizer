import os
from pathlib import Path
from typing import List

from .metadata import extract_metadata
from .video import extract_frames
from .llm import LLMDescriber


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


def gather_files(directory: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                files.append(os.path.join(root, name))
    return files


def describe_and_rename(path: str, llm: LLMDescriber) -> None:
    info_lines = []
    ext = Path(path).suffix.lower()
    metadata = extract_metadata(path)
    info_lines.append(f"Modified time: {metadata.modified_time}")
    if metadata.gps:
        info_lines.append(f"GPS: {metadata.gps}")
    if metadata.camera_model:
        info_lines.append(f"Camera: {metadata.camera_model}")

    if ext in VIDEO_EXTS:
        frames_dir = os.path.join(os.path.dirname(path), f"frames_{Path(path).stem}")
        frames = extract_frames(path, frames_dir)
        info_lines.append(f"Extracted {len(frames)} frames")

    info_text = "\n".join(info_lines)
    description = llm.describe(info_text)

    new_name = description.short_name + Path(path).suffix
    new_path = os.path.join(os.path.dirname(path), new_name)
    os.rename(path, new_path)
    print(f"Renamed {path} -> {new_path}")


def main(directory: str):
    llm = LLMDescriber()
    files = gather_files(directory)
    for path in files:
        describe_and_rename(path, llm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize memories using an LLM")
    parser.add_argument("directory", help="Directory with photos and videos")
    args = parser.parse_args()
    main(args.directory)
