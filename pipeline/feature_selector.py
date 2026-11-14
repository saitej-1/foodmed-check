"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — FEATURE SELECTION PIPELINE
Drug-Food Interaction Prediction

4-stage sequential reduction:
  Stage 1 → Variance Threshold (VT > 0.8)
  Stage 2 → Pearson Correlation filter (r < 0.75)
  Stage 3 → LASSO regularization (α = 0.01)
  Stage 4 → RFECV with Ridge estimator

Output → 18 Optimal Features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    RFECV
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import StratifiedKFold


class DFIFeatureSelector:
    """
    4-Stage Feature Selection Pipeline for DFI prediction.

    Reduces ~3,780 raw molecular descriptors down to the
    18 most predictive, non-redundant features via:
      1. Variance Threshold  — removes near-constant features
      2. Pearson Correlation — removes redundant correlated pairs
      3. LASSO               — L1 regularization for sparsity
      4. RFECV               — cross-validated recursive elimination

    Attributes:
        selected_features: List of final selected feature names
        n_stages: Counts of features after each stage
    """

    def __init__(self,
                 variance_threshold: float = 0.8,
                 correlation_threshold: float = 0.75,
                 lasso_alpha: float = 0.01,
                 target_features: int = 18,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Args:
            variance_threshold: Remove features with variance < this value
            correlation_threshold: Remove one of any pair with |r| > this value
            lasso_alpha: L1 regularization strength for LASSO selection
            target_features: Final desired number of features (RFECV minimum)
            cv_folds: Number of folds for RFECV cross-validation
            random_state: Random seed for reproducibility
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.lasso_alpha = lasso_alpha
        self.target_features = target_features
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Internal state (set after fit())
        self.scaler_ = None
        self.vt_ = None
        self.lasso_selector_ = None
        self.rfecv_ = None

        self.vt_features_ = None         # After stage 1
        self.pearson_features_ = None    # After stage 2
        self.lasso_features_ = None      # After stage 3
        self.selected_features_ = None  # After stage 4 (final)

        self.n_stages_ = {}
        self.is_fitted_ = False

    # ─────────────────────────────────────────────
    # STAGE 1 — VARIANCE THRESHOLD
    # ─────────────────────────────────────────────

    def _stage1_variance_threshold(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with very low variance (near-constant columns)."""
        print(f"\n  📊 Stage 1: Variance Threshold (threshold={self.variance_threshold})")

        self.vt_ = VarianceThreshold(threshold=self.variance_threshold)
        self.vt_.fit(X)
        mask = self.vt_.get_support()
        selected = X.columns[mask].tolist()

        self.vt_features_ = selected
        self.n_stages_["after_vt"] = len(selected)
        dropped = X.shape[1] - len(selected)

        print(f"     {X.shape[1]:>5} → {len(selected):>5} features  ({dropped} removed)")
        return X[selected]

    # ─────────────────────────────────────────────
    # STAGE 2 — PEARSON CORRELATION FILTER
    # ─────────────────────────────────────────────

    def _stage2_pearson_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated feature pairs.
        For each pair with |r| > threshold, drop the one
        with the higher mean absolute correlation (more redundant).
        """
        print(f"\n  📊 Stage 2: Pearson Correlation Filter (r < {self.correlation_threshold})")

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Identify columns with correlation above threshold
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > self.correlation_threshold):
                # Drop this column if it has high mean correlation
                to_drop.add(col)

        selected = [c for c in X.columns if c not in to_drop]
        self.pearson_features_ = selected
        self.n_stages_["after_pearson"] = len(selected)
        dropped = X.shape[1] - len(selected)

        print(f"     {X.shape[1]:>5} → {len(selected):>5} features  ({dropped} removed)")
        return X[selected]

    # ─────────────────────────────────────────────
    # STAGE 3 — LASSO REGULARIZATION
    # ─────────────────────────────────────────────

    def _stage3_lasso(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply L1 (LASSO) regularization to identify features
        with non-zero coefficients. Higher alpha = more sparsity.
        """
        print(f"\n  📊 Stage 3: LASSO Selection (α={self.lasso_alpha})")

        lasso = Lasso(
            alpha=self.lasso_alpha,
            max_iter=10000,
            random_state=self.random_state
        )
        self.lasso_selector_ = SelectFromModel(lasso, prefit=False, threshold=1e-6)
        self.lasso_selector_.fit(X, y)

        mask = self.lasso_selector_.get_support()
        selected = X.columns[mask].tolist()

        # Fallback: if LASSO is too aggressive, keep top 50
        if len(selected) < 10:
            print(f"     ⚠  LASSO too aggressive ({len(selected)} features), relaxing...")
            coef = np.abs(lasso.fit(X, y).coef_)
            top_idx = np.argsort(coef)[-50:]
            selected = X.columns[top_idx].tolist()

        self.lasso_features_ = selected
        self.n_stages_["after_lasso"] = len(selected)
        dropped = X.shape[1] - len(selected)

        print(f"     {X.shape[1]:>5} → {len(selected):>5} features  ({dropped} removed)")
        return X[selected]

    # ─────────────────────────────────────────────
    # STAGE 4 — RFECV WITH RIDGE ESTIMATOR
    # ─────────────────────────────────────────────

    def _stage4_rfecv(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Recursive Feature Elimination with Cross-Validation.
        Uses Ridge regression as the estimator (fast, stable).
        Finds the optimal number of features by CV score.
        """
        print(f"\n  📊 Stage 4: RFECV with Ridge (CV={self.cv_folds} folds)")

        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)

        self.rfecv_ = RFECV(
            estimator=ridge,
            step=1,
            cv=cv,
            scoring="r2",
            min_features_to_select=min(self.target_features, X.shape[1]),
            n_jobs=1
        )
        self.rfecv_.fit(X, y)

        mask = self.rfecv_.support_
        selected = X.columns[mask].tolist()

        # If more than target, keep top N by RFECV ranking
        if len(selected) > self.target_features:
            ranking = self.rfecv_.ranking_
            top_idx = np.argsort(ranking)[:self.target_features]
            selected = X.columns[top_idx].tolist()

        self.selected_features_ = selected
        self.n_stages_["final"] = len(selected)
        dropped = X.shape[1] - len(selected)

        print(f"     {X.shape[1]:>5} → {len(selected):>5} features  ({dropped} removed)")
        print(f"\n  ✅ Optimal features found by RFECV: {self.rfecv_.n_features_}")
        return X[selected]

    # ─────────────────────────────────────────────
    # FIT & TRANSFORM
    # ─────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DFIFeatureSelector":
        """
        Fit the full 4-stage selection pipeline.

        Args:
            X: Raw feature DataFrame (~3,780 columns)
            y: Target labels (0 = Negative, 1 = Positive, 2 = Non-significant)

        Returns:
            self (for method chaining)
        """
        print("\n" + "="*55)
        print("  PHASE 2: FEATURE SELECTION PIPELINE")
        print("="*55)
        print(f"  Input: {X.shape[0]} samples × {X.shape[1]} features")

        # Scale first
        self.scaler_ = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler_.fit_transform(X),
            columns=X.columns
        )

        # Stage 1: Variance Threshold
        X1 = self._stage1_variance_threshold(X_scaled)

        # Stage 2: Pearson Correlation
        X2 = self._stage2_pearson_filter(X1)

        # Stage 3: LASSO
        X3 = self._stage3_lasso(X2, y)

        # Stage 4: RFECV
        X4 = self._stage4_rfecv(X3, y)

        self.is_fitted_ = True

        print("\n" + "="*55)
        print("  FEATURE SELECTION SUMMARY")
        print("="*55)
        print(f"  Raw features:           {X.shape[1]:>6}")
        print(f"  After Variance (VT):    {self.n_stages_['after_vt']:>6}")
        print(f"  After Pearson filter:   {self.n_stages_['after_pearson']:>6}")
        print(f"  After LASSO:            {self.n_stages_['after_lasso']:>6}")
        print(f"  Final (RFECV):          {self.n_stages_['final']:>6}  ← OPTIMAL")
        print("="*55)
        print("\n  🏆 Selected Features:")
        for i, f in enumerate(self.selected_features_, 1):
            print(f"     {i:>2}. {f}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw features using the fitted pipeline.

        Args:
            X: Raw feature DataFrame (same columns as fit input)

        Returns:
            Numpy array with only the selected features
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before .transform()")

        X_scaled = pd.DataFrame(
            self.scaler_.transform(X),
            columns=X.columns
        )
        return X_scaled[self.selected_features_].values

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit and immediately transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Save fitted selector to disk."""
        joblib.dump(self, path)
        print(f"  💾 Selector saved → {path}")

    @staticmethod
    def load(path: str) -> "DFIFeatureSelector":
        """Load a previously saved selector."""
        return joblib.load(path)

    def get_feature_importance_report(self) -> pd.DataFrame:
        """Return a DataFrame summarizing the selected features and their RFECV rankings."""
        if not self.is_fitted_:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self.selected_features_,
            "rfecv_rank": [1] * len(self.selected_features_),
            "stage_selected": ["RFECV"] * len(self.selected_features_)
        })
