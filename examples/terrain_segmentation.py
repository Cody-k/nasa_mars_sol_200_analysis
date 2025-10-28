"""Terrain segmentation demo | Analyze Mars surface composition"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import MastcamImageLoader, TerrainSegmenter
from src.vision.visualize_terrain import (
    plot_segmentation_overlay,
    plot_composition_summary,
    plot_rock_detection,
    plot_texture_map,
)


def main():
    """Segment and analyze terrain in Mastcam images"""

    print("=== Mars Terrain Segmentation Analysis ===\n")

    mastcam_dir = Path("data/raw/mastcam")
    output_dir = Path("output/terrain_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = MastcamImageLoader(mastcam_dir)
    segmenter = TerrainSegmenter()

    print("Loading images...")
    images = loader.load_all_mastcam()
    print(f"Loaded {len(images)} images\n")

    print("Segmenting terrain...")
    compositions = []
    filenames = []

    for i, img_data in enumerate(images[:10], 1):
        result = segmenter.segment_terrain(img_data["data"])
        compositions.append(result["composition"])
        filenames.append(img_data["filename"])

        comp = result["composition"]
        print(f"[{i}] {img_data['filename'][:30]}")
        print(f"    Sky: {comp['sky']:.1f}%  Rocks: {comp['rocks']:.1f}%  Sand: {comp['sand']:.1f}%")

        if i <= 3:
            output_path = output_dir / f"segmentation_{i:02d}.png"
            plot_segmentation_overlay(
                img_data["data"],
                result["sky_mask"],
                result["rock_mask"],
                result["sand_mask"],
                output_path,
            )
            print(f"    Saved: {output_path}")

    print("\nOverall statistics:")
    avg_sky = sum(c["sky"] for c in compositions) / len(compositions)
    avg_rocks = sum(c["rocks"] for c in compositions) / len(compositions)
    avg_sand = sum(c["sand"] for c in compositions) / len(compositions)
    print(f"Average sky coverage: {avg_sky:.1f}%")
    print(f"Average rock coverage: {avg_rocks:.1f}%")
    print(f"Average sand coverage: {avg_sand:.1f}%")

    print("\nGenerating composition summary...")
    summary_path = output_dir / "composition_summary.png"
    plot_composition_summary(compositions, filenames, summary_path)
    print(f"Saved: {summary_path}")

    print("\nAnalyzing individual rocks in first image...")
    first_image = images[0]["data"]
    result = segmenter.segment_terrain(first_image)
    rocks = segmenter.detect_individual_rocks(result["rock_mask"])
    print(f"Detected {len(rocks)} rock objects")

    if rocks:
        print(f"Largest rock: {rocks[0]['area']} pixels")
        print(f"Top 5 rock sizes: {[r['area'] for r in rocks[:5]]}")

        rock_path = output_dir / "rock_detection.png"
        plot_rock_detection(first_image, rocks, rock_path, max_rocks=20)
        print(f"Saved: {rock_path}")

    print("\nGenerating texture map...")
    texture_map = segmenter.calculate_texture_map(
        segmenter._to_grayscale(first_image)
    )
    texture_path = output_dir / "texture_map.png"
    plot_texture_map(first_image, texture_map, texture_path)
    print(f"Saved: {texture_path}")

    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
