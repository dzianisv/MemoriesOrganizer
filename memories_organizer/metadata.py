import os
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


@dataclass
class FileMetadata:
    path: str
    modified_time: float
    gps: Optional[Tuple[float, float]] = None
    camera_model: Optional[str] = None


def _get_exif_data(image: Image.Image) -> dict:
    exif_data = {}
    info = image._getexif()
    if not info:
        return exif_data
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_data[decoded] = value
    return exif_data


def _get_gps(exif_data: dict) -> Optional[Tuple[float, float]]:
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None
    gps_data = {}
    for key in gps_info:
        decode = GPSTAGS.get(key, key)
        gps_data[decode] = gps_info[key]
    try:
        lat = gps_data.get("GPSLatitude")
        lat_ref = gps_data.get("GPSLatitudeRef")
        lon = gps_data.get("GPSLongitude")
        lon_ref = gps_data.get("GPSLongitudeRef")
        if not all([lat, lon, lat_ref, lon_ref]):
            return None
        lat = _convert_to_degrees(lat)
        lon = _convert_to_degrees(lon)
        if lat_ref != "N":
            lat = -lat
        if lon_ref != "E":
            lon = -lon
        return lat, lon
    except Exception:
        return None


def _convert_to_degrees(value):
    d0, d1 = value[0]
    d = d0 / d1
    m0, m1 = value[1]
    m = m0 / m1
    s0, s1 = value[2]
    s = s0 / s1
    return d + (m / 60.0) + (s / 3600.0)


def extract_metadata(path: str) -> FileMetadata:
    modified_time = os.path.getmtime(path)
    gps = None
    camera_model = None
    try:
        with Image.open(path) as img:
            exif_data = _get_exif_data(img)
            gps = _get_gps(exif_data)
            camera_model = exif_data.get("Model")
    except Exception:
        # Not an image or failed to read EXIF
        pass

    return FileMetadata(path=path, modified_time=modified_time, gps=gps, camera_model=camera_model)
