"""Mastcam analysis | Load and analyze Sol 200 surface images"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import MastcamImageLoader


def main():
    """Analyze Mastcam images from Sol 200"""

    print("=== Mastcam Image Analysis ===\n")

    mastcam_dir = Path("data/raw/mastcam")
    loader = MastcamImageLoader(mastcam_dir)

    print("Loading Mastcam images...")
    images = loader.load_all_mastcam()

    print(f"Loaded {len(images)} images\n")

    for i, img_data in enumerate(images[:5], 1):
        print(f"[{i}] {img_data['filename']}")
        print(f"    Shape: {img_data['shape']}")
        print(f"    Data type: {img_data['dtype']}")

        stats = loader.get_statistics(img_data['data'])
        print(f"    Range: {stats['min']:.0f} - {stats['max']:.0f}")
        print(f"    Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")

    if images:
        print(f"\n... and {len(images) - 5} more images")

        print("\nSaving first image as PNG for visualization...")
        output = Path("output/mastcam_example.png")
        output.parent.mkdir(exist_ok=True)

        loader.save_as_png(images[0]['data'], output)
        print(f"Saved: {output}")

    print("\n=== Analysis Complete ===")
    print(f"Total images: {len(images)}")
    print("Mastcam images ready for computer vision analysis (segmentation, feature detection)")


if __name__ == "__main__":
    main()
