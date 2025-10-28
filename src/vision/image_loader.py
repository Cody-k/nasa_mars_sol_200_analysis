"""Mastcam image loader | Load and process NASA PDS IMG files"""

from pathlib import Path
from typing import Optional
import numpy as np
import struct

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


class MastcamImageLoader:
    """Load Mastcam images from NASA PDS IMG format"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_image(self, img_file: Path) -> Optional[np.ndarray]:
        """
        Load PDS IMG file to numpy array

        Mastcam images are typically 16-bit unsigned integers
        """
        try:
            label_file = img_file.with_suffix(".lbl")

            if not label_file.exists():
                return None

            dimensions = self._parse_label(label_file)

            if not dimensions:
                return None

            lines = dimensions.get("LINES", 0)
            line_samples = dimensions.get("LINE_SAMPLES", 0)

            with open(img_file, "rb") as f:
                data = f.read()

            image_array = np.frombuffer(data, dtype=">u2")
            image_array = image_array.reshape((lines, line_samples))

            return image_array
        except Exception:
            return None

    def _parse_label(self, label_file: Path) -> Optional[dict]:
        """Parse PDS label file for image dimensions"""
        try:
            content = label_file.read_text()
            dimensions = {}

            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("LINES") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        dimensions["LINES"] = int(val)
                    except ValueError:
                        pass

                if line.startswith("LINE_SAMPLES") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        dimensions["LINE_SAMPLES"] = int(val)
                    except ValueError:
                        pass

            return dimensions if "LINES" in dimensions and "LINE_SAMPLES" in dimensions else None
        except Exception:
            return None

    def load_all_mastcam(self) -> list[dict]:
        """Load all Mastcam images from data directory"""
        img_files = sorted(self.data_dir.glob("*.img"))

        images = []
        for img_file in img_files:
            array = self.load_image(img_file)

            if array is not None:
                images.append({
                    "filename": img_file.name,
                    "data": array,
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                })

        return images

    def get_statistics(self, image_array: np.ndarray) -> dict:
        """Calculate basic image statistics"""
        return {
            "shape": image_array.shape,
            "min": float(image_array.min()),
            "max": float(image_array.max()),
            "mean": float(image_array.mean()),
            "std": float(image_array.std()),
        }

    def normalize_for_display(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image for display (0-255 range)"""
        normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        return (normalized * 255).astype(np.uint8)

    def save_as_png(self, image_array: np.ndarray, output_path: Path) -> None:
        """Save image array as PNG"""
        if not PILLOW_AVAILABLE:
            raise ImportError("Pillow required for PNG export")

        normalized = self.normalize_for_display(image_array)
        img = Image.fromarray(normalized)
        img.save(output_path)
