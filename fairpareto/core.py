"""
Core FairParetoClassifier implementation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from aequitas.flow.utils import LabeledFrame

from .calibration import train_calibrated_clf
from .optimizer import EvelPertoFront
from .utils import equal_frequency_binning


class FairParetoClassifier(BaseEstimator):
    """
    Fair binary classifier that computes the optimal fairness-performance Pareto front.

    This classifier implements the MIFPO (Model Independent Fairness-Performance
    Optimization) algorithm to compute the theoretical optimal trade-off between
    fairness (measured by statistical parity) and performance (accuracy) for binary
    classification tasks.

    Parameters
    ----------
    sensitive_column : str, optional
        Name of the sensitive attribute column in X. Required for tabular data mode.

    strategy : {'histogram', 'eqz_bins', 'exact'}, default='histogram'
        Binning strategy for probability discretization:
        - 'histogram': uniform width bins
        - 'eqz_bins': equal frequency bins
        - 'exact': use exact probability values (no binning)

    n_bins : int, default=10
        Number of bins for probability discretization (ignored if strategy='exact').

    k : int, default=2
        MIFPO optimization parameter controlling representation complexity.

    verbose : bool, default=False
        Whether to print progress information during fitting.

    Attributes
    ----------
    pareto_front_ : dict
        Dictionary mapping fairness levels (gamma) to optimal accuracy values.
        Available after calling fit().
    """

    def __init__(self, sensitive_column=None, strategy='histogram', n_bins=10, k=2,gamma_values=None, verbose=False):
        self.sensitive_column = sensitive_column
        self.strategy = strategy
        self.n_bins = n_bins
        self.k = k
        self.verbose = verbose
        self.gamma_values = gamma_values

    def fit(self, X, y, calibrated_classifier=None):
        """
        Fit the classifier and compute the fairness-performance Pareto front.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training data. Must include the sensitive attribute column if
            sensitive_column is specified.

        y : array-like of shape (n_samples,)
            Binary target values (0 or 1).

        calibrated_classifier : object, optional
            Pre-trained calibrated classifier with predict_proba method.
            If None, an internal XGBoost classifier will be trained and calibrated.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.sensitive_column is None:
            raise ValueError("sensitive_column must be specified for tabular data. "
                             "Use fit_presplit() for pre-split data.")

        if self.verbose:
            print("FairPareto: Starting fit process...")
            print(f"  - Data shape: {X.shape}")
            print(f"  - Strategy: {self.strategy}, n_bins: {self.n_bins}, k: {self.k}")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if self.verbose:
                print("  - Converting array to DataFrame...")
            X = pd.DataFrame(X)
            if self.sensitive_column not in X.columns:
                raise ValueError(f"sensitive_column '{self.sensitive_column}' not found in data")

        # Check sensitive column exists
        if self.sensitive_column not in X.columns:
            raise ValueError(f"sensitive_column '{self.sensitive_column}' not found in data")

        # Create dataset in expected format
        if self.verbose:
            print("  - Preparing dataset...")

        data = X.copy()
        data['label'] = y
        data['attr'] = data[self.sensitive_column]

        # Remove the original sensitive column to avoid duplication
        data = data.drop(columns=[self.sensitive_column])

        # Create dataset object (using only train data, no splitting)
        train_data = LabeledFrame(data, y_col='label', s_col='attr')

        # Split data by sensitive attribute
        X_0 = train_data.X[train_data.s == 0]
        X_1 = train_data.X[train_data.s == 1]
        y_0 = train_data.y[train_data.s == 0]
        y_1 = train_data.y[train_data.s == 1]

        if self.verbose:
            print(f"  - Group 0 size: {len(X_0)}, Group 1 size: {len(X_1)}")

        # Train or use provided calibrated classifiers
        if calibrated_classifier is None:
            if self.verbose:
                print("  - Training calibrated classifiers...")
            clf_0 = train_calibrated_clf(X_0, y_0)
            clf_1 = train_calibrated_clf(X_1, y_1)
        else:
            if self.verbose:
                print("  - Using provided calibrated classifier...")
            clf_0 = calibrated_classifier
            clf_1 = calibrated_classifier

        # Compute Pareto front
        if self.verbose:
            print("  - Computing Pareto front...")

        self.pareto_front_ = self._compute_pareto_front(
            X_0, X_1, y_0, y_1, clf_0, clf_1,gamma_values = self.gamma_values
        )

        if self.verbose:
            print(f"  - Pareto front computed with {len(self.pareto_front_)} points")
            print("FairPareto: Fit complete!")

        return self

    def fit_presplit(self, X_0, X_1, y_0, y_1, clf_0, clf_1):
        """
        Fit the classifier using pre-split data and calibrated classifiers.

        This method is intended for non-tabular data (e.g., images, text) where
        the data has already been split by sensitive attribute and calibrated
        classifiers have been trained for each group.

        Parameters
        ----------
        X_0 : array-like of shape (n_samples_0, n_features)
            Training data for sensitive group 0.

        X_1 : array-like of shape (n_samples_1, n_features)
            Training data for sensitive group 1.

        y_0 : array-like of shape (n_samples_0,)
            Binary target values for sensitive group 0.

        y_1 : array-like of shape (n_samples_1,)
            Binary target values for sensitive group 1.

        clf_0 : object
            Trained calibrated classifier for sensitive group 0.
            Must have predict_proba method.

        clf_1 : object
            Trained calibrated classifier for sensitive group 1.
            Must have predict_proba method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.verbose:
            print("FairPareto: Starting fit_presplit process...")
            print(f"  - Group 0 shape: {X_0.shape}, Group 1 shape: {X_1.shape}")
            print(f"  - Strategy: {self.strategy}, n_bins: {self.n_bins}, k: {self.k}")
            print("  - Computing Pareto front...")

        self.pareto_front_ = self._compute_pareto_front(
            X_0, X_1, y_0, y_1, clf_0, clf_1
        )

        if self.verbose:
            print(f"  - Pareto front computed with {len(self.pareto_front_)} points")
            print("FairPareto: Fit complete!")

        return self

    def _compute_pareto_front(self, X_0, X_1, y_0, y_1, clf_0, clf_1, gamma_values = None):
        """
        Internal method to compute the Pareto front.
        """
        # Get probability predictions
        if hasattr(clf_0, 'predict_proba'):
            probs_0 = clf_0.predict_proba(X_0)[:, 1]
            probs_1 = clf_1.predict_proba(X_1)[:, 1]
        else:
            raise ValueError("Classifiers must have predict_proba method")

        # Apply binning strategy
        if self.verbose:
            print(f"    - Applying {self.strategy} binning...")

        U, pu, lu = self._apply_binning_strategy(probs_1, self.n_bins, self.strategy)
        V, pv, lv = self._apply_binning_strategy(probs_0, self.n_bins, self.strategy)

        # Setup optimization parameters
        n, m = U.shape[0], V.shape[0]
        alpha = lu / (lu + lv)

        if self.verbose:
            print(f"    - Optimization parameters: n={n}, m={m}, alpha={alpha:.3f}")

        # Create and run optimizer
        opt_problem = EvelPertoFront(n, m, self.k, alpha, pu, pv, U, V, loss="ACC",gamma_values = gamma_values)

        # Define gamma values for Pareto front
        gamma_values = self.gamma_values if self.gamma_values is not None else list(np.linspace(0, 1, 31))

        if self.verbose:
            print(f"    - Running optimization for {len(gamma_values)} gamma values...")

        opt_problem.run(sorted(gamma_values))

        return opt_problem.get_results()

    def _apply_binning_strategy(self, probabilities, n_bins, strategy):
        """
        Apply the specified binning strategy to probability values.
        """
        if strategy == "histogram":
            _f_bined, bins = np.histogram(probabilities, n_bins)
            bins_centers = (bins[:-1] + bins[1:]) / 2
            f_bined = _f_bined / sum(_f_bined)
            return f_bined.reshape(-1, 1), bins_centers.reshape(-1, 1), len(probabilities)

        elif strategy == "eqz_bins":
            return equal_frequency_binning(probabilities, n_bins)

        elif strategy == "exact":
            bins_centers = np.array(list(set(sorted(probabilities))))
            f_bined = np.ones_like(bins_centers) * (1 / bins_centers.shape[0])
            return f_bined.reshape(-1, 1), bins_centers.reshape(-1, 1), len(probabilities)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_pareto_front(self):
        """
        Get the computed fairness-performance Pareto front.

        Returns
        -------
        dict
            Dictionary mapping fairness levels (gamma) to optimal accuracy values.

        Raises
        ------
        ValueError
            If the classifier has not been fitted yet.
        """
        if not hasattr(self, 'pareto_front_'):
            raise ValueError("Classifier must be fitted before accessing Pareto front. "
                             "Call fit() or fit_presplit() first.")
        return self.pareto_front_

    def plot_pareto_front(self, title="Fairness-Performance Pareto Front", ax=None):
        """
        Plot the computed Pareto front.

        Parameters
        ----------
        title : str, default="Fairness-Performance Pareto Front"
            Title for the plot.

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        if not hasattr(self, 'pareto_front_'):
            raise ValueError("Classifier must be fitted before plotting. "
                             "Call fit() or fit_presplit() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. "
                              "Install it with: pip install matplotlib")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        gammas = list(self.pareto_front_.keys())
        accuracies = list(self.pareto_front_.values())

        ax.plot(gammas, accuracies, 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Fairness Level (γ - Statistical Parity)')
        ax.set_ylabel('Optimal Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax