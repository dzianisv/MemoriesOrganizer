from pathlib import Path
import numpy as np
from PIL import Image
import piexif
import cv2

from memories_organizer.metadata import extract_metadata
from memories_organizer.video import extract_frames
from memories_organizer.main import gather_files, describe_and_rename
from memories_organizer.llm import Description


def _create_sample_image(path: Path) -> None:
    img = Image.new("RGB", (10, 10), color="blue")
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
    exif_dict["0th"][piexif.ImageIFD.Model] = "TestCam"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = [(40, 1), (0, 1), (0, 1)]
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E"
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = [(70, 1), (0, 1), (0, 1)]
    exif_bytes = piexif.dump(exif_dict)
    img.save(path, exif=exif_bytes)


def _create_sample_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 30, (32, 32))
    for i in range(60):
        frame = np.full((32, 32, 3), i % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_metadata_extraction(tmp_path):
    img_path = tmp_path / "photo.jpg"
    _create_sample_image(img_path)
    meta = extract_metadata(str(img_path))
    assert meta.gps == (40.0, 70.0)
    assert meta.camera_model == "TestCam"


def test_frame_extraction(tmp_path):
    video_path = tmp_path / "video.mp4"
    _create_sample_video(video_path)
    frames_dir = tmp_path / "frames"
    frames = extract_frames(str(video_path), str(frames_dir), every_seconds=1)
    assert len(frames) == 2
    for frame in frames:
        assert Path(frame).exists()


def test_describe_and_rename(tmp_path):
    img_path = tmp_path / "photo.jpg"
    _create_sample_image(img_path)

    files = gather_files(str(tmp_path))
    assert str(img_path) in files

    class FakeLLM:
        def describe(self, info: str) -> Description:
            return Description(description="desc", short_name="renamed")

    llm = FakeLLM()
    describe_and_rename(str(img_path), llm)
    assert (tmp_path / "renamed.jpg").exists()
