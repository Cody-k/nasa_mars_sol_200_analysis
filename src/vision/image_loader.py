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

        Handles both 8-bit RGB and 16-bit grayscale formats
        """
        try:
            label_file = img_file.with_suffix(".lbl")

            if not label_file.exists():
                return None

            params = self._parse_label(label_file)

            if not params:
                return None

            lines = params.get("LINES", 0)
            line_samples = params.get("LINE_SAMPLES", 0)
            bands = params.get("BANDS", 1)
            sample_bits = params.get("SAMPLE_BITS", 8)

            with open(img_file, "rb") as f:
                data = f.read()

            if sample_bits == 8:
                dtype = np.uint8
            elif sample_bits == 16:
                dtype = ">u2"
            else:
                return None

            image_array = np.frombuffer(data, dtype=dtype)

            if bands == 3:
                image_array = image_array.reshape((bands, lines, line_samples))
                image_array = np.transpose(image_array, (1, 2, 0))
            else:
                image_array = image_array.reshape((lines, line_samples))

            return image_array
        except Exception:
            return None

    def _parse_label(self, label_file: Path) -> Optional[dict]:
        """Parse PDS label file for image parameters"""
        try:
            content = label_file.read_text()
            params = {}

            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("LINES") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        params["LINES"] = int(val)
                    except ValueError:
                        pass

                if line.startswith("LINE_SAMPLES") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        params["LINE_SAMPLES"] = int(val)
                    except ValueError:
                        pass

                if line.startswith("BANDS") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        params["BANDS"] = int(val)
                    except ValueError:
                        pass

                if line.startswith("SAMPLE_BITS") and "=" in line:
                    try:
                        val = line.split("=")[1].strip()
                        params["SAMPLE_BITS"] = int(val)
                    except ValueError:
                        pass

            return params if "LINES" in params and "LINE_SAMPLES" in params else None
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
