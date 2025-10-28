"""Particle classification demo | Train ML classifier for radiation particle ID"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.radiation.pha_parser import PHAParser
from src.radiation.particle_classifier import ParticleClassifier
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Train and evaluate particle classification model"""

    print("=== Mars Radiation Particle Classification ===\n")

    rad_file = Path("data/raw/RAD_RDR_2013_058_02_42_0200_V00.TXT")

    print("Parsing PHA events from RAD data...")
    parser = PHAParser()
    pha_df = parser.parse_pha_events(rad_file)
    print(f"Parsed {len(pha_df)} PHA events")

    print("\nFiltering valid events...")
    pha_df = parser.get_valid_events(pha_df)
    print(f"Valid events: {len(pha_df)}")

    print("\nCalculating total energy...")
    pha_df = parser.calculate_total_energy(pha_df)

    print("Calculating detector counts...")
    pha_df = parser.calculate_detector_counts(pha_df)

    print(f"\nEnergy range: {pha_df['total_energy'].min():.2f} - {pha_df['total_energy'].max():.2f} MeV")
    print(f"Mean detectors triggered: {pha_df['n_detectors_triggered'].mean():.1f}")

    print("\n=== Training Particle Classifier ===\n")

    classifier = ParticleClassifier()

    print("Engineering physics-informed features...")
    pha_df = classifier.engineer_features(pha_df)

    print("Applying physics-based particle labels...")
    pha_df = classifier.label_particles_physics(pha_df)

    particle_counts = pha_df.group_by("particle_type").len()
    print("\nPhysics-labeled particle distribution:")
    for row in particle_counts.iter_rows(named=True):
        pct = (row["len"] / len(pha_df)) * 100
        print(f"  {row['particle_type']}: {row['len']} ({pct:.1f}%)")

    labeled_df = pha_df.filter(pha_df["particle_type"] != "unknown")
    print(f"\nLabeled events for training: {len(labeled_df)}")

    if len(labeled_df) < 50:
        print("\nInsufficient labeled data for training. Need at least 50 events.")
        return

    print("\nPreparing training data...")
    X, y = classifier.prepare_training_data(labeled_df)

    print(f"Training samples: {len(X)}")
    print(f"Features: {len(classifier.feature_cols)}")

    print("\nTraining XGBoost classifier...")
    results = classifier.train(X, y)

    print(f"\nTraining accuracy: {results['train_accuracy']:.3f}")
    print(f"Test accuracy: {results['test_accuracy']:.3f}")

    print("\nPer-class metrics:")
    for cls in classifier.classes:
        if cls in results["classification_report"]:
            metrics = results["classification_report"][cls]
            print(f"  {cls}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-score: {metrics['f1-score']:.3f}")
            print(f"    Support: {int(metrics['support'])}")

    print("\nFeature importance:")
    importance = classifier.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"  {feat}: {imp:.3f}")

    print("\nGenerating visualizations...")
    output_dir = Path("output/particle_classification")
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_confusion_matrix(results["confusion_matrix"], classifier.classes, output_dir)
    _plot_feature_importance(importance, output_dir)
    _plot_particle_distributions(labeled_df, output_dir)

    print("\n=== Classification Complete ===")
    print(f"Results saved to: {output_dir}/")


def _plot_confusion_matrix(cm, classes, output_dir):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i][j], ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Particle Classification Confusion Matrix")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/confusion_matrix.png")


def _plot_feature_importance(importance, output_dir):
    """Plot feature importance"""
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_features)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color="steelblue")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance for Particle Classification")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/feature_importance.png")


def _plot_particle_distributions(df, output_dir):
    """Plot energy and detector distributions by particle type"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    particle_types = ["proton", "alpha", "heavy_ion", "electron"]
    colors = ["blue", "red", "green", "orange"]

    for ptype, color in zip(particle_types, colors):
        subset = df.filter(df["particle_type"] == ptype)
        if len(subset) > 0:
            energies = subset["total_energy"].to_numpy()
            axes[0, 0].hist(energies, bins=30, alpha=0.5, label=ptype, color=color)

    axes[0, 0].set_xlabel("Total Energy (MeV)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Energy Distribution by Particle Type")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    for ptype, color in zip(particle_types, colors):
        subset = df.filter(df["particle_type"] == ptype)
        if len(subset) > 0:
            counts = subset["n_detectors_triggered"].to_numpy()
            axes[0, 1].hist(counts, bins=range(0, 37), alpha=0.5, label=ptype, color=color)

    axes[0, 1].set_xlabel("Number of Detectors Triggered")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Detector Counts by Particle Type")
    axes[0, 1].legend()

    for ptype, color in zip(particle_types, colors):
        subset = df.filter(df["particle_type"] == ptype)
        if len(subset) > 0:
            ratios = subset["ratio_back_front"].to_numpy()
            axes[1, 0].hist(ratios, bins=30, alpha=0.5, label=ptype, color=color)

    axes[1, 0].set_xlabel("Back/Front Energy Ratio")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Energy Ratio Distribution")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 10)

    particle_counts = df.group_by("particle_type").len()
    types = [row["particle_type"] for row in particle_counts.iter_rows(named=True)]
    counts = [row["len"] for row in particle_counts.iter_rows(named=True)]

    axes[1, 1].bar(types, counts, color=colors[:len(types)])
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Particle Type Distribution")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "particle_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/particle_distributions.png")


if __name__ == "__main__":
    main()
