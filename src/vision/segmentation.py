"""Terrain segmentation | Classical CV for Mars surface analysis"""

from typing import Optional
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure


class TerrainSegmenter:
    """Segment Mars terrain into sky, rocks, and sand regions"""

    def __init__(self):
        self.sky_threshold = 0.7
        self.rock_min_size = 100

    def segment_terrain(self, image: np.ndarray) -> dict:
        """
        Segment image into terrain components

        Returns dict with:
        - sky_mask: Binary mask of sky region
        - rock_mask: Binary mask of rock regions
        - sand_mask: Binary mask of sand/soil regions
        - composition: Percentage coverage of each class
        """
        gray = self._to_grayscale(image)

        sky_mask = self._detect_sky(gray)
        ground_mask = ~sky_mask

        rock_mask, sand_mask = self._separate_rocks_sand(gray, ground_mask)

        total_pixels = gray.size
        composition = {
            "sky": float(np.sum(sky_mask) / total_pixels * 100),
            "rocks": float(np.sum(rock_mask) / total_pixels * 100),
            "sand": float(np.sum(sand_mask) / total_pixels * 100),
        }

        return {
            "sky_mask": sky_mask,
            "rock_mask": rock_mask,
            "sand_mask": sand_mask,
            "composition": composition,
        }

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if image.ndim == 3:
            return np.mean(image, axis=2).astype(np.uint8)
        return image

    def _detect_sky(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect sky region using gradient and intensity

        Sky typically appears in upper portion with:
        - High intensity (bright)
        - Low texture (smooth gradients)
        """
        normalized = gray.astype(float) / 255.0

        height = gray.shape[0]
        horizon_region = int(height * 0.4)

        gradient_y = np.abs(ndimage.sobel(normalized, axis=0))

        sky_mask = np.zeros_like(gray, dtype=bool)

        for row in range(horizon_region):
            row_intensity = normalized[row].mean()
            row_gradient = gradient_y[row].mean()

            if row_intensity > self.sky_threshold and row_gradient < 0.15:
                sky_mask[row] = True

        sky_mask = ndimage.binary_fill_holes(sky_mask)
        sky_mask = morphology.binary_opening(sky_mask, morphology.disk(3))

        return sky_mask

    def _separate_rocks_sand(
        self, gray: np.ndarray, ground_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate rocks from sand using texture and intensity

        Rocks: Darker regions with high local variance
        Sand: Uniform regions with low texture
        """
        ground_only = gray.copy()
        ground_only[~ground_mask] = 0

        local_variance = ndimage.generic_filter(
            ground_only.astype(float), np.var, size=11
        )

        texture_threshold = filters.threshold_otsu(
            local_variance[ground_mask].flatten()
        )

        high_texture = local_variance > texture_threshold
        rock_candidates = high_texture & ground_mask

        rock_mask = morphology.remove_small_objects(
            rock_candidates, min_size=self.rock_min_size
        )

        rock_mask = morphology.binary_closing(rock_mask, morphology.disk(3))

        sand_mask = ground_mask & ~rock_mask

        return rock_mask, sand_mask

    def detect_individual_rocks(self, rock_mask: np.ndarray) -> list[dict]:
        """
        Identify individual rock objects

        Returns list of rock properties:
        - area: Number of pixels
        - centroid: (y, x) coordinates
        - bbox: (min_row, min_col, max_row, max_col)
        - eccentricity: Shape metric (0=circle, 1=line)
        """
        labeled = measure.label(rock_mask)
        regions = measure.regionprops(labeled)

        rocks = []
        for region in regions:
            rocks.append(
                {
                    "area": region.area,
                    "centroid": region.centroid,
                    "bbox": region.bbox,
                    "eccentricity": region.eccentricity,
                }
            )

        return sorted(rocks, key=lambda r: r["area"], reverse=True)

    def calculate_texture_map(
        self, gray: np.ndarray, window_size: int = 15
    ) -> np.ndarray:
        """
        Calculate local texture strength

        Higher values indicate more texture (rocks, features)
        Lower values indicate smooth regions (sand, sky)
        """
        local_std = ndimage.generic_filter(gray.astype(float), np.std, size=window_size)

        return local_std

    def estimate_horizon_line(self, sky_mask: np.ndarray) -> Optional[int]:
        """
        Estimate horizon line position

        Returns row index of approximate horizon (sky/ground boundary)
        """
        if not np.any(sky_mask):
            return None

        col_transitions = []

        for col in range(sky_mask.shape[1]):
            sky_column = sky_mask[:, col]

            if np.any(sky_column):
                last_sky_pixel = np.where(sky_column)[0][-1]
                col_transitions.append(last_sky_pixel)

        if not col_transitions:
            return None

        return int(np.median(col_transitions))
