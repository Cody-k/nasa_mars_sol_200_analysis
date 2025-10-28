"""Particle classifier | ML-based identification of radiation particle species"""

from typing import Optional
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class ParticleClassifier:
    """
    Train ML classifier to identify particle species from PHA data

    Uses physics-based rules to generate training labels, then trains
    XGBoost classifier for automated particle identification
    """

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.feature_cols: list[str] = []
        self.classes = ["proton", "alpha", "heavy_ion", "electron"]

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create physics-informed features from raw PHA data

        Features:
        - Energy ratios between detector layers
        - Total energy deposited
        - Number of detectors triggered
        - Energy distribution patterns
        - Penetration depth indicators
        """
        df = self._calculate_layer_energies(df)
        df = self._calculate_energy_ratios(df)
        df = self._calculate_spatial_patterns(df)

        return df

    def _calculate_layer_energies(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate energy by detector layer (front, middle, back)"""
        front_detectors = [f"corr_{i:02d}" for i in range(0, 12)]
        middle_detectors = [f"corr_{i:02d}" for i in range(12, 24)]
        back_detectors = [f"corr_{i:02d}" for i in range(24, 36)]

        df = df.with_columns([
            pl.sum_horizontal([
                pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(0.0)
                for c in front_detectors
            ]).alias("energy_front"),
            pl.sum_horizontal([
                pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(0.0)
                for c in middle_detectors
            ]).alias("energy_middle"),
            pl.sum_horizontal([
                pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(0.0)
                for c in back_detectors
            ]).alias("energy_back"),
        ])

        return df

    def _calculate_energy_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate characteristic energy ratios

        - back/front ratio: Distinguishes stopping vs penetrating particles
        - middle/front ratio: Penetration depth indicator
        - Energy asymmetry: Spatial distribution metric
        """
        df = df.with_columns([
            (pl.col("energy_back") / (pl.col("energy_front") + 1e-6))
            .alias("ratio_back_front"),
            (pl.col("energy_middle") / (pl.col("energy_front") + 1e-6))
            .alias("ratio_middle_front"),
            (pl.col("energy_back") / (pl.col("energy_middle") + 1e-6))
            .alias("ratio_back_middle"),
        ])

        df = df.with_columns(
            ((pl.col("energy_front") - pl.col("energy_back")).abs() /
             (pl.col("total_energy") + 1e-6))
            .alias("energy_asymmetry")
        )

        return df

    def _calculate_spatial_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate spatial distribution metrics"""
        all_detectors = [f"corr_{i:02d}" for i in range(36)]

        df = df.with_columns([
            pl.max_horizontal([pl.col(c) for c in all_detectors]).alias("max_channel"),
            pl.min_horizontal([
                pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(pl.lit(float("inf")))
                for c in all_detectors
            ]).alias("min_positive_channel"),
        ])

        df = df.with_columns(
            (pl.col("max_channel") / (pl.col("min_positive_channel") + 1e-6))
            .alias("channel_dynamic_range")
        )

        return df

    def label_particles_physics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply physics-based rules to label particle species

        Rules based on energy deposition patterns:
        - Protons: Moderate energy, back/front ratio 2-4, most common
        - Alphas: Higher energy, back/front ratio 1-2, ~10% of events
        - Heavy ions: Very high energy, saturates detectors, rare
        - Electrons: Low energy, front-dominant, low penetration
        """
        conditions = [
            (
                (pl.col("total_energy") > 10.0) &
                (pl.col("max_channel") > 15.0) &
                (pl.col("n_detectors_triggered") >= 5)
            ).alias("is_heavy_ion"),
            (
                (pl.col("ratio_back_front") >= 1.0) &
                (pl.col("ratio_back_front") <= 2.5) &
                (pl.col("total_energy") >= 2.0) &
                (pl.col("n_detectors_triggered") >= 3)
            ).alias("is_alpha"),
            (
                (pl.col("ratio_back_front") >= 2.0) &
                (pl.col("ratio_back_front") <= 5.0) &
                (pl.col("total_energy") >= 0.5) &
                (pl.col("total_energy") <= 15.0)
            ).alias("is_proton"),
            (
                (pl.col("energy_front") > pl.col("energy_back")) &
                (pl.col("total_energy") < 2.0) &
                (pl.col("n_detectors_triggered") <= 3)
            ).alias("is_electron"),
        ]

        df = df.with_columns(conditions)

        df = df.with_columns(
            pl.when(pl.col("is_heavy_ion"))
            .then(pl.lit("heavy_ion"))
            .when(pl.col("is_alpha"))
            .then(pl.lit("alpha"))
            .when(pl.col("is_proton"))
            .then(pl.lit("proton"))
            .when(pl.col("is_electron"))
            .then(pl.lit("electron"))
            .otherwise(pl.lit("unknown"))
            .alias("particle_type")
        )

        df = df.drop(["is_heavy_ion", "is_alpha", "is_proton", "is_electron"])

        return df

    def prepare_training_data(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for ML training

        Returns (X, y) arrays for scikit-learn/XGBoost
        """
        feature_cols = [
            "total_energy",
            "n_detectors_triggered",
            "energy_front",
            "energy_middle",
            "energy_back",
            "ratio_back_front",
            "ratio_middle_front",
            "ratio_back_middle",
            "energy_asymmetry",
            "max_channel",
            "channel_dynamic_range",
        ]

        self.feature_cols = feature_cols

        X = df.select(feature_cols).to_numpy()

        label_map = {
            "proton": 0,
            "alpha": 1,
            "heavy_ion": 2,
            "electron": 3,
        }

        y = df.select("particle_type").to_series().map_elements(
            lambda x: label_map.get(x, -1), return_dtype=pl.Int32
        ).to_numpy()

        valid_mask = y >= 0
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Train XGBoost classifier

        Returns training metrics and evaluation results
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
        )

        self.model.fit(X_train, y_train)

        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        y_pred = self.model.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred)

        return {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def get_feature_importance(self) -> dict:
        """
        Get feature importance from trained model

        Returns dict mapping feature names to importance scores
        """
        if self.model is None:
            return {}

        importances = self.model.feature_importances_

        return dict(zip(self.feature_cols, importances.tolist()))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict particle types for new data

        Adds 'predicted_particle' column to DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = df.select(self.feature_cols).to_numpy()

        predictions = self.model.predict(X)

        class_names = [self.classes[p] for p in predictions]

        df = df.with_columns(pl.Series("predicted_particle", class_names))

        return df
