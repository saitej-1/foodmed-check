"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — MODEL TRAINING & PREDICTION
Drug-Food Interaction Prediction

Trains an XGBoost multiclass classifier with:
  - Stratified K-Fold cross-validation
  - Bayesian hyperparameter optimization
  - Class weighting for imbalanced data

Prediction Classes:
  Class 0 → Negative DFI (avoid completely)
  Class 1 → Positive DFI (safe to take together)
  Class 2 → Non-significant DFI (monitor / use caution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score,
    cross_validate, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb


# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────

CLASS_LABELS = {
    0: "Negative DFI",
    1: "Positive DFI",
    2: "Non-significant DFI"
}

CLASS_COLORS = {
    0: "🔴",
    1: "🟢",
    2: "🟡"
}


class DFIModelTrainer:
    """
    Phase 3: XGBoost Classifier for Drug-Food Interaction prediction.

    Handles:
      - Automatic hyperparameter search (RandomizedSearchCV)
      - Stratified cross-validation
      - Class weight balancing
      - Full evaluation report generation
      - Model persistence (save/load)

    Usage:
        trainer = DFIModelTrainer()
        trainer.fit(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)
        proba = trainer.predict_proba(X_new)
    """

    def __init__(self,
                 n_classes: int = 3,
                 cv_folds: int = 5,
                 tune_hyperparams: bool = True,
                 n_search_iter: int = 50,
                 random_state: int = 42):
        """
        Args:
            n_classes: Number of output classes (3 for DFI)
            cv_folds: Folds for stratified cross-validation
            tune_hyperparams: Run RandomizedSearchCV if True, else use defaults
            n_search_iter: Number of hyperparameter combinations to try
            random_state: Random seed for reproducibility
        """
        self.n_classes = n_classes
        self.cv_folds = cv_folds
        self.tune_hyperparams = tune_hyperparams
        self.n_search_iter = n_search_iter
        self.random_state = random_state

        self.model_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.feature_importances_ = None
        self.is_fitted_ = False

    # ─────────────────────────────────────────────
    # HYPERPARAMETER SPACE
    # ─────────────────────────────────────────────

    def _get_param_grid(self) -> dict:
        """XGBoost hyperparameter search space."""
        return {
            "n_estimators":       [100, 200, 300, 400, 500],
            "max_depth":          [3, 4, 5, 6, 7, 8],
            "learning_rate":      [0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight":   [1, 3, 5, 7],
            "gamma":              [0, 0.1, 0.2, 0.3, 0.5],
            "reg_alpha":          [0, 0.01, 0.1, 1.0],    # L1
            "reg_lambda":         [0.5, 1.0, 2.0, 5.0],   # L2
        }

    def _get_default_params(self) -> dict:
        """Default high-performing XGBoost params (no search)."""
        return {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────

    def _build_model(self, params: dict, class_weights: np.ndarray | None = None) -> xgb.XGBClassifier:
        """Instantiate XGBoost classifier with given params."""
        base_params = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            **params
        }
        return xgb.XGBClassifier(**base_params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "DFIModelTrainer":
        """
        Train the XGBoost classifier with optional hyperparameter tuning.

        Args:
            X: Feature matrix (n_samples × n_features)
            y: Target labels (0, 1, or 2)
            X_val: Optional validation set for early stopping
            y_val: Optional validation labels

        Returns:
            self
        """
        print("\n" + "="*55)
        print("  PHASE 3: MODEL TRAINING")
        print("="*55)
        print(f"  Training samples: {X.shape[0]}")
        print(f"  Features:         {X.shape[1]}")
        print(f"  Classes:          {np.unique(y).tolist()}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\n  Class distribution:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y) * 100
            print(f"    {CLASS_COLORS.get(cls,'')} Class {cls} ({CLASS_LABELS.get(cls,'?')}): {cnt} ({pct:.1f}%)")

        # Compute class weights for imbalanced data
        unique_classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=unique_classes, y=y)
        class_to_weight = {cls: w for cls, w in zip(unique_classes, weights)}
        sample_weight = np.array([class_to_weight[int(yi)] for yi in y])

        # ── Hyperparameter Search ──
        if self.tune_hyperparams:
            print(f"\n  🔍 Hyperparameter Search ({self.n_search_iter} iterations, {self.cv_folds}-fold CV)...")
            base_model = self._build_model({})
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                  random_state=self.random_state)
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self._get_param_grid(),
                n_iter=self.n_search_iter,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=1,
                random_state=self.random_state,
                verbose=1
            )
            search.fit(X, y, sample_weight=sample_weight)
            self.best_params_ = search.best_params_
            print(f"\n  ✅ Best params found:")
            for k, v in self.best_params_.items():
                print(f"     {k}: {v}")
        else:
            self.best_params_ = self._get_default_params()
            print("\n  Using default hyperparameters (tune_hyperparams=False)")

        # ── Final Model Training ──
        print("\n  🚀 Training final model...")
        self.model_ = self._build_model(self.best_params_)

        fit_kwargs = {"sample_weight": sample_weight}
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = 20
            fit_kwargs["verbose"] = False

        self.model_.fit(X, y, **fit_kwargs)

        # ── Cross-Validation Scores ──
        print(f"\n  📈 Running {self.cv_folds}-fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)
        cv_scores = cross_validate(
            self.model_, X, y, cv=cv,
            scoring=["accuracy", "f1_weighted"],
            n_jobs=-1
        )
        self.cv_results_ = {
            "accuracy": cv_scores["test_accuracy"],
            "f1_weighted": cv_scores["test_f1_weighted"],
        }

        print(f"\n  Cross-Validation Results ({self.cv_folds} folds):")
        print(f"    Accuracy   : {np.mean(cv_scores['test_accuracy']):.4f} ± {np.std(cv_scores['test_accuracy']):.4f}")
        print(f"    F1-weighted: {np.mean(cv_scores['test_f1_weighted']):.4f} ± {np.std(cv_scores['test_f1_weighted']):.4f}")

        # Feature importances
        self.feature_importances_ = self.model_.feature_importances_
        self.is_fitted_ = True

        print("\n  ✅ Model training complete!")
        return self

    # ─────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0, 1, or 2)."""
        self._check_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns:
            Array of shape (n_samples, 3) with probabilities for
            [Negative DFI, Positive DFI, Non-significant DFI]
        """
        self._check_fitted()
        return self.model_.predict_proba(X)

    def predict_single(self, x: np.ndarray) -> dict:
        """
        Full prediction for a single feature vector.

        Returns:
            Dictionary with predicted class, probabilities, and label
        """
        self._check_fitted()
        x = x.reshape(1, -1)
        cls = int(self.model_.predict(x)[0])
        proba = self.model_.predict_proba(x)[0]
        return {
            "predicted_class": cls,
            "predicted_label": CLASS_LABELS[cls],
            "confidence": float(proba[cls]),
            "probabilities": {
                CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)
            }
        }

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Generate a comprehensive evaluation report on test data.

        Returns:
            Dictionary with all evaluation metrics
        """
        self._check_fitted()

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        # AUC-ROC (one vs rest)
        try:
            y_bin = label_binarize(y_test, classes=[0, 1, 2])
            auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        print("\n" + "="*55)
        print("  PHASE 3: EVALUATION REPORT")
        print("="*55)
        print(f"  Test Accuracy:    {acc:.4f}")
        print(f"  F1 (weighted):    {f1:.4f}")
        if auc:
            print(f"  AUC-ROC (OvR):    {auc:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[CLASS_LABELS[i] for i in range(self.n_classes)]
        ))
        print(f"\n  Confusion Matrix:")
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {CLASS_LABELS[i]}" for i in range(self.n_classes)],
            columns=[f"Pred {CLASS_LABELS[i]}" for i in range(self.n_classes)]
        )
        print(cm_df.to_string())
        print("="*55)

        return {
            "accuracy": acc,
            "f1_weighted": f1,
            "auc_roc": auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[CLASS_LABELS[i] for i in range(self.n_classes)],
                output_dict=True
            ),
            "cv_accuracy_mean": float(np.mean(self.cv_results_["accuracy"])) if self.cv_results_ else None,
            "cv_f1_mean": float(np.mean(self.cv_results_["f1_weighted"])) if self.cv_results_ else None,
        }

    def get_top_features(self, feature_names: list[str], top_n: int = 18) -> pd.DataFrame:
        """Return top N features by XGBoost importance score."""
        self._check_fitted()
        if self.feature_importances_ is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": feature_names[:len(self.feature_importances_)],
            "importance": self.feature_importances_
        }).sort_values("importance", ascending=False)
        return df.head(top_n)

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the entire trainer to disk."""
        joblib.dump(self, path)
        print(f"  💾 Trainer saved → {path}")

    @staticmethod
    def load(path: str) -> "DFIModelTrainer":
        """Load a saved trainer."""
        return joblib.load(path)

    def export_model_card(self, output_path: str) -> None:
        """Export a JSON model card with all training metadata."""
        card = {
            "model": "XGBoost Multiclass Classifier",
            "task": "Drug-Food Interaction Prediction",
            "classes": CLASS_LABELS,
            "best_params": self.best_params_,
            "cv_results": {
                "accuracy_mean": float(np.mean(self.cv_results_["accuracy"])),
                "accuracy_std": float(np.std(self.cv_results_["accuracy"])),
                "f1_mean": float(np.mean(self.cv_results_["f1_weighted"])),
                "f1_std": float(np.std(self.cv_results_["f1_weighted"])),
            } if self.cv_results_ else {},
            "n_features": len(self.feature_importances_) if self.feature_importances_ is not None else None,
        }
        with open(output_path, "w") as f:
            json.dump(card, f, indent=2)
        print(f"  📋 Model card saved → {output_path}")

    def _check_fitted(self):
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Model not trained yet. Call .fit() first.")
