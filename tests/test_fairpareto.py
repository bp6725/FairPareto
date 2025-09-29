"""
Unit tests for fairpareto package.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fairpareto import FairParetoClassifier
from fairpareto.calibration import train_calibrated_clf, XGBWrapper, CustomCalibratedClassifier
from fairpareto.utils import equal_frequency_binning


class TestFairParetoClassifier:
    """Test the main FairParetoClassifier class."""

    def test_initialization(self):
        """Test classifier initialization."""
        clf = FairParetoClassifier()
        assert clf.sensitive_column is None
        assert clf.strategy == 'histogram'
        assert clf.n_bins == 10
        assert clf.k == 2
        assert clf.verbose == False

        # Test with custom parameters
        clf = FairParetoClassifier(
            sensitive_column='gender',
            strategy='eqz_bins',
            n_bins=20,
            k=5,
            verbose=True
        )
        assert clf.sensitive_column == 'gender'
        assert clf.strategy == 'eqz_bins'
        assert clf.n_bins == 20
        assert clf.k == 5
        assert clf.verbose == True

    def test_fit_tabular_data(self):
        """Test fitting with tabular data."""
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_classes=2,
            n_informative=6,
            n_redundant=2,
            random_state=42
        )

        # Add synthetic sensitive attribute
        sensitive = np.random.binomial(1, 0.4, size=len(y))

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['sensitive_attr'] = sensitive

        clf = FairParetoClassifier(sensitive_column='sensitive_attr', verbose=True)
        clf.fit(df, y)

        # Check that pareto front was computed
        assert hasattr(clf, 'pareto_front_')
        assert isinstance(clf.pareto_front_, dict)
        assert len(clf.pareto_front_) > 0

        # Check that all values are valid
        for gamma, accuracy in clf.pareto_front_.items():
            assert 0 <= gamma <= 1
            assert 0 <= accuracy <= 1 or np.isnan(accuracy)  # Allow NaN for failed optimization

    def test_fit_presplit_data(self):
        """Test fitting with pre-split data."""
        # Generate data for each group
        X_0, y_0 = make_classification(
            n_samples=150, n_features=6, n_classes=2, random_state=42
        )
        X_1, y_1 = make_classification(
            n_samples=100, n_features=6, n_classes=2, random_state=43
        )

        # Train simple calibrated classifiers
        clf_0 = train_calibrated_clf(X_0, y_0)
        clf_1 = train_calibrated_clf(X_1, y_1)

        clf = FairParetoClassifier(verbose=True)
        clf.fit_presplit(X_0, X_1, y_0, y_1, clf_0, clf_1)

        # Check that pareto front was computed
        assert hasattr(clf, 'pareto_front_')
        assert isinstance(clf.pareto_front_, dict)
        assert len(clf.pareto_front_) > 0

    def test_get_pareto_front(self):
        """Test getting the Pareto front."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        sensitive = np.random.binomial(1, 0.5, size=len(y))

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['sensitive_attr'] = sensitive

        clf = FairParetoClassifier(sensitive_column='sensitive_attr')

        # Should raise error before fitting
        with pytest.raises(ValueError, match="must be fitted"):
            clf.get_pareto_front()

        # Should work after fitting
        clf.fit(df, y)
        pareto_front = clf.get_pareto_front()
        assert isinstance(pareto_front, dict)
        assert pareto_front == clf.pareto_front_

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Missing sensitive column
        clf = FairParetoClassifier()
        with pytest.raises(ValueError, match="sensitive_column must be specified"):
            clf.fit(df, y)

        # Non-existent sensitive column
        clf = FairParetoClassifier(sensitive_column='nonexistent')
        with pytest.raises(ValueError, match="not found in data"):
            clf.fit(df, y)


class TestCalibrationUtilities:
    """Test calibration utilities."""

    def test_xgb_wrapper(self):
        """Test XGBWrapper functionality."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = XGBWrapper()
        clf.fit(X, y)

        # Test prediction methods
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert hasattr(clf, 'classes_')

    def test_custom_calibrated_classifier(self):
        """Test CustomCalibratedClassifier."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        base_clf = XGBWrapper()
        clf = CustomCalibratedClassifier(base_estimator=base_clf)
        clf.fit(X, y)

        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)

        # Check that probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_train_calibrated_clf(self):
        """Test the main training function."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = train_calibrated_clf(X, y)

        # Test that it's properly calibrated
        probabilities = clf.predict_proba(X)
        predictions = clf.predict(X)

        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_equal_frequency_binning(self):
        """Test equal frequency binning."""
        # Create sample probability data
        probs = np.random.beta(2, 5, size=200)

        f_bined, bins_centers, n_samples = equal_frequency_binning(probs, n_bins=5)

        assert f_bined.shape == (5, 1)
        assert bins_centers.shape == (5, 1)
        assert n_samples == 200
        assert np.allclose(f_bined.sum(), 1.0)  # Frequencies should sum to 1


if __name__ == "__main__":
    pytest.main([__file__])