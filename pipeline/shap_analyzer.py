"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4A — SHAP EXPLAINABILITY
Drug-Food Interaction Prediction

Uses SHAP (SHapley Additive exPlanations) to explain why
the XGBoost model made each prediction.

Outputs:
  - Feature importance bar plots
  - SHAP summary beeswarm plots
  - Per-prediction waterfall explanations
  - Force plots for individual decisions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed. Run: pip install shap")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Class metadata
CLASS_LABELS = {
    0: "Negative DFI\n(Avoid Together)",
    1: "Positive DFI\n(Safe Together)",
    2: "Non-significant DFI\n(Use Caution)"
}

CLASS_COLORS_MPL = ["#ef4444", "#22c55e", "#f59e0b"]  # red, green, amber


class DFISHAPAnalyzer:
    """
    Phase 4A: SHAP Analysis for Drug-Food Interaction explanations.

    Computes SHAP values for each prediction, explaining which
    molecular features drove the model's decision.

    Usage:
        analyzer = DFISHAPAnalyzer(trainer.model_)
        analyzer.fit(X_train)
        explanation = analyzer.explain_prediction(x_new, feature_names)
        analyzer.plot_summary(X_test, feature_names)
    """

    def __init__(self, xgb_model):
        """
        Args:
            xgb_model: Fitted XGBoost model from DFIModelTrainer.model_
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = xgb_model
        self.explainer_ = None
        self.shap_values_ = None
        self.base_values_ = None
        self.is_fitted_ = False

    def fit(self, X_background: np.ndarray = None) -> "DFISHAPAnalyzer":
        """
        Initialize TreeExplainer.

        Args:
            X_background: Optional training data used as SHAP baseline (typically X_train).
        """
        print("\n  🔬 Initializing SHAP TreeExplainer...")
        try:
            if X_background is not None:
                self.explainer_ = shap.TreeExplainer(
                    self.model,
                    data=X_background[:min(200, len(X_background))],
                    feature_perturbation="tree_path_dependent"
                )
                # Warm-up call to test if it raises Categorical split or other exceptions
                _ = self.explainer_(X_background[:1])
            else:
                self.explainer_ = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"  ⚠ TreeExplainer initialization or warm-up failed: {e}")
            print("  Falling back to tree-path dependent explainer without background data.")
            self.explainer_ = shap.TreeExplainer(self.model)

        self.is_fitted_ = True
        print("  ✅ SHAP explainer ready")
        return self

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Returns:
            SHAP values array of shape (n_samples, n_features, n_classes)
        """
        self._check_fitted()
        print(f"  Computing SHAP values for {X.shape[0]} samples...")
        self.shap_values_ = self.explainer_(X)
        return self.shap_values_

    # ─────────────────────────────────────────────
    # PER-PREDICTION EXPLANATION
    # ─────────────────────────────────────────────

    def explain_prediction(self,
                           x: np.ndarray,
                           feature_names: list[str],
                           top_n: int = 10) -> dict:
        """
        Explain a single prediction with SHAP values.

        Args:
            x: Single feature vector (1D array)
            feature_names: List of feature names
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with SHAP explanation and top features
        """
        self._check_fitted()
        x = x.reshape(1, -1)

        # Predicted class
        pred_class = int(self.model.predict(x)[0])
        pred_proba = self.model.predict_proba(x)[0]

        # Compute SHAP for this sample
        shap_exp = self.explainer_(x)

        # SHAP values for the predicted class
        sv = shap_exp.values[0, :, pred_class]

        # Top contributing features
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        top_features = []
        for i in sorted_idx[:top_n]:
            top_features.append({
                "feature": feature_names[i] if i < len(feature_names) else f"feat_{i}",
                "shap_value": float(sv[i]),
                "direction": "increases risk" if sv[i] > 0 else "decreases risk",
                "feature_value": float(x[0, i])
            })

        return {
            "predicted_class": pred_class,
            "predicted_label": CLASS_LABELS[pred_class],
            "confidence": float(pred_proba[pred_class]),
            "probabilities": {
                f"Class {i}": float(p) for i, p in enumerate(pred_proba)
            },
            "top_contributing_features": top_features,
            "base_value": float(shap_exp.base_values[0, pred_class]),
            "total_shap_contribution": float(np.sum(np.abs(sv)))
        }

    # ─────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────

    def plot_summary(self,
                     X: np.ndarray,
                     feature_names: list[str],
                     class_idx: int = 0,
                     save_path: str | None = None) -> None:
        """
        SHAP summary beeswarm plot — shows global feature importance
        and direction of effect for a given class.

        Args:
            X: Feature matrix
            feature_names: Feature names
            class_idx: Which class to explain (0=Negative, 1=Positive, 2=Non-sig)
            save_path: Optional path to save figure
        """
        if not PLOT_AVAILABLE:
            print("⚠  matplotlib not available for plotting")
            return

        shap_vals = self.compute_shap_values(X)

        plt.figure(figsize=(12, 8))
        plt.title(
            f"SHAP Summary — {CLASS_LABELS[class_idx]}\n"
            "Feature Impact on DFI Prediction",
            fontsize=14, fontweight="bold", pad=15
        )

        shap.summary_plot(
            shap_vals.values[:, :, class_idx],
            X,
            feature_names=feature_names,
            show=False,
            plot_type="beeswarm",
            max_display=18
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Summary plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_importance(self,
                                X: np.ndarray,
                                feature_names: list[str],
                                save_path: str | None = None) -> None:
        """
        Bar chart of mean |SHAP| values across all classes.
        Shows global feature importance.
        """
        if not PLOT_AVAILABLE:
            return

        shap_vals = self.compute_shap_values(X)
        # Mean absolute SHAP across all classes and samples
        mean_abs = np.mean(np.abs(shap_vals.values), axis=(0, 2))
        sorted_idx = np.argsort(mean_abs)[-18:]  # Top 18

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_idx)))

        bars = ax.barh(
            [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in sorted_idx],
            mean_abs[sorted_idx],
            color=colors[::-1],
            edgecolor="white",
            linewidth=0.5
        )

        ax.set_xlabel("Mean |SHAP Value| (Global Feature Importance)", fontsize=12)
        ax.set_title("Top 18 Most Important Features\n(SHAP Analysis — Drug-Food Interaction Model)",
                     fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Importance plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_waterfall(self,
                       x: np.ndarray,
                       feature_names: list[str],
                       class_idx: int = 0,
                       save_path: str | None = None) -> None:
        """
        Waterfall plot for a single prediction — shows contribution
        of each feature from baseline to final prediction.
        """
        if not PLOT_AVAILABLE:
            return

        x = x.reshape(1, -1)
        shap_exp = self.explainer_(x)

        plt.figure(figsize=(12, 6))
        plt.title(
            f"SHAP Waterfall — {CLASS_LABELS[class_idx]}\n"
            "How each feature contributed to this prediction",
            fontsize=13, fontweight="bold"
        )

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_exp.values[0, :, class_idx],
                base_values=shap_exp.base_values[0, class_idx],
                data=x[0],
                feature_names=feature_names
            ),
            show=False
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Waterfall plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_all_classes_importance(self,
                                    X: np.ndarray,
                                    feature_names: list[str],
                                    save_path: str | None = None) -> None:
        """
        Side-by-side bar charts of SHAP importance for each class.
        """
        if not PLOT_AVAILABLE:
            return

        shap_vals = self.compute_shap_values(X)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle("SHAP Feature Importance per Interaction Class",
                     fontsize=14, fontweight="bold", y=1.02)

        for cls_idx, ax in enumerate(axes):
            mean_abs = np.mean(np.abs(shap_vals.values[:, :, cls_idx]), axis=0)
            sorted_idx = np.argsort(mean_abs)[-12:]
            names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in sorted_idx]

            ax.barh(names, mean_abs[sorted_idx], color=CLASS_COLORS_MPL[cls_idx],
                    alpha=0.85, edgecolor="white")
            ax.set_title(CLASS_LABELS[cls_idx], fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean |SHAP|")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📊 All-class SHAP plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(X_train) before using the explainer.")
