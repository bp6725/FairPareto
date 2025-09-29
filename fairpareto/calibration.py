"""
Calibrated classifier utilities for fairpareto.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class CustomCalibratedClassifier:
    """
    Custom calibrated classifier wrapper using isotonic regression.

    This class wraps any base estimator with isotonic calibration to ensure
    that predict_proba outputs represent true probabilities.

    Parameters
    ----------
    base_estimator : object
        Base classifier to calibrate. Must implement fit and predict_proba.
    """

    def __init__(self, base_estimator):
        self.calcls = CalibratedClassifierCV(
            base_estimator=base_estimator, method="isotonic"
        )

    def __sklearn_tags__(self):
        return {
            "requires_positive_X": False,
            "X_types": ["2darray"],
            "requires_y": True,
        }

    def fit(self, X, y):
        """
        Fit the calibrated classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self.calcls.fit(X, y)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return self.modify_probas(self.calcls.predict_proba(X))

    def modify_probas(self, probas):
        """
        Hook for modifying probabilities. Override in subclasses if needed.

        Parameters
        ----------
        probas : array-like
            Raw probabilities from the calibrated classifier.

        Returns
        -------
        array-like
            Modified probabilities.
        """
        return probas

class XGBWrapper(BaseEstimator, ClassifierMixin):
    """
    XGBoost wrapper with automatic hyperparameter tuning.

    This wrapper automatically performs grid search to find optimal
    hyperparameters for the XGBoost classifier.
    """

    def __init__(self):
        self.params = {}
        self.model = XGBClassifier(**self.params)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """
        Fit XGBoost with automatic hyperparameter tuning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert torch tensors to numpy if needed
        if hasattr(X, 'cpu'):
            X = X.cpu().numpy()
        if hasattr(y, 'cpu'):
            y = y.cpu().numpy()

        # Define parameter grid for grid search
        param_grid = {
            "n_estimators": [50, 150],
            "learning_rate": [0.001, 0.01, 0.1],
            "gamma": [0, 0.8],
            "max_depth": [2, 5],
            "subsample": [0.5, 1.0],
            "reg_lambda": [0.5, 1, 3],
        }

        # Perform grid search
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X, y)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        if hasattr(X, 'cpu'):
            X = X.cpu().numpy()
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if hasattr(X, 'cpu'):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)

    def __sklearn_tags__(self):
        return {
            "requires_positive_X": False,
            "X_types": ["2darray"],
            "requires_y": True,
            "allow_nan": True,
        }

def train_calibrated_clf(train, targets):
    """
    Train a calibrated XGBoost classifier.

    This function creates and trains a calibrated XGBoost classifier
    using the XGBWrapper and CustomCalibratedClassifier.

    Parameters
    ----------
    train : array-like of shape (n_samples, n_features)
        Training data.
    targets : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    CustomCalibratedClassifier
        Trained and calibrated classifier.
    """
    clf = XGBWrapper()
    custom_calcls = CustomCalibratedClassifier(base_estimator=clf)
    custom_calcls.fit(train, targets)
    return custom_calcls