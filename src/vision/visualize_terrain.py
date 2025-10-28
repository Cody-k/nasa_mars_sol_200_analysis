"""Terrain visualization | Display segmentation results"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_segmentation_overlay(
    image: np.ndarray,
    sky_mask: np.ndarray,
    rock_mask: np.ndarray,
    sand_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    Visualize segmentation with color overlay

    Sky: Blue overlay
    Rocks: Red overlay
    Sand: Yellow overlay
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    gray = _to_grayscale(image)

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    overlay = np.zeros((*gray.shape, 4))

    overlay[sky_mask] = [0.3, 0.6, 1.0, 0.4]
    overlay[rock_mask] = [1.0, 0.2, 0.2, 0.5]
    overlay[sand_mask] = [1.0, 0.9, 0.3, 0.3]

    axes[1].imshow(gray, cmap="gray")
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation (Blue=Sky, Red=Rocks, Yellow=Sand)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_composition_summary(
    compositions: list[dict], filenames: list[str], output_path: Path
) -> None:
    """
    Plot surface composition statistics across multiple images

    Shows distribution of sky, rocks, and sand percentages
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sky_pcts = [c["sky"] for c in compositions]
    rock_pcts = [c["rocks"] for c in compositions]
    sand_pcts = [c["sand"] for c in compositions]

    axes[0, 0].hist(sky_pcts, bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Sky Coverage (%)")
    axes[0, 0].set_ylabel("Number of Images")
    axes[0, 0].set_title("Sky Coverage Distribution")
    axes[0, 0].axvline(np.mean(sky_pcts), color="red", linestyle="--", label="Mean")
    axes[0, 0].legend()

    axes[0, 1].hist(rock_pcts, bins=15, color="coral", edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Rock Coverage (%)")
    axes[0, 1].set_ylabel("Number of Images")
    axes[0, 1].set_title("Rock Coverage Distribution")
    axes[0, 1].axvline(np.mean(rock_pcts), color="red", linestyle="--", label="Mean")
    axes[0, 1].legend()

    axes[1, 0].hist(sand_pcts, bins=15, color="gold", edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Sand/Soil Coverage (%)")
    axes[1, 0].set_ylabel("Number of Images")
    axes[1, 0].set_title("Sand Coverage Distribution")
    axes[1, 0].axvline(np.mean(sand_pcts), color="red", linestyle="--", label="Mean")
    axes[1, 0].legend()

    x = np.arange(len(compositions))
    width = 0.25

    axes[1, 1].bar(x - width, sky_pcts, width, label="Sky", color="skyblue", alpha=0.8)
    axes[1, 1].bar(x, rock_pcts, width, label="Rocks", color="coral", alpha=0.8)
    axes[1, 1].bar(
        x + width, sand_pcts, width, label="Sand", color="gold", alpha=0.8
    )

    axes[1, 1].set_xlabel("Image Index")
    axes[1, 1].set_ylabel("Coverage (%)")
    axes[1, 1].set_title("Surface Composition by Image")
    axes[1, 1].legend()
    axes[1, 1].set_xticks(x[::5])
    axes[1, 1].set_xticklabels(x[::5])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rock_detection(
    image: np.ndarray, rocks: list[dict], output_path: Path, max_rocks: int = 20
) -> None:
    """
    Visualize detected individual rocks with bounding boxes

    Shows largest rocks with labels
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    gray = _to_grayscale(image)
    ax.imshow(gray, cmap="gray")

    for i, rock in enumerate(rocks[:max_rocks], 1):
        min_row, min_col, max_row, max_col = rock["bbox"]

        width = max_col - min_col
        height = max_row - min_row

        rect = Rectangle(
            (min_col, min_row),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(rect)

        centroid_y, centroid_x = rock["centroid"]
        ax.plot(centroid_x, centroid_y, "r+", markersize=12, markeredgewidth=2)

        if i <= 10:
            ax.text(
                min_col,
                min_row - 10,
                f"#{i}",
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_title(f"Rock Detection (Top {min(max_rocks, len(rocks))} Objects)", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_texture_map(
    image: np.ndarray, texture_map: np.ndarray, output_path: Path
) -> None:
    """
    Visualize texture strength across image

    Highlights high-texture regions (rocks, features)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    gray = _to_grayscale(image)

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(texture_map, cmap="hot")
    axes[1].set_title("Texture Map (Bright = High Texture)", fontsize=14)
    axes[1].axis("off")

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Local Std Dev", rotation=270, labelpad=20)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed"""
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.uint8)
    return image
