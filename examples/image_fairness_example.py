"""
Image fairness example for FairPareto package.

This example demonstrates how to use FairPareto with image data using
pre-split groups and custom CNN classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

from fairpareto.core import FairParetoClassifier


from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleImageClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple sklearn-compatible image classifier that mimics a CNN for demonstration.

    In practice, you would use actual CNNs like ResNet, VGG, etc.
    This is a simplified version using sklearn's MLPClassifier.
    """

    def __init__(self, image_shape=(28, 28), n_classes=2, hidden_layer_sizes=(128, 64),
                 max_iter=200, random_state=42):
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

        # Calculate input size
        self.input_size = np.prod(image_shape)

        # Use MLPClassifier as a simple neural network
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'image_shape': self.image_shape,
            'n_classes': self.n_classes,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)

        # Reinitialize the model with new parameters
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        return self

    def fit(self, X, y):
        """Fit the classifier to flattened image data."""
        # Flatten images if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        self.model.fit(X_flat, y)
        self.classes_ = np.unique(y)  # Required for sklearn compatibility
        return self

    def predict_proba(self, X):
        """Predict probabilities for image data."""
        # Flatten images if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        return self.model.predict_proba(X_flat)

    def predict(self, X):
        """Predict class labels for image data."""
        # Flatten images if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        return self.model.predict(X_flat)


def create_synthetic_image_data(n_samples_group_0=400, n_samples_group_1=300,
                               image_shape=(28, 28), bias_strength=0.3):
    """
    Create synthetic image-like data split by sensitive groups.

    In practice, you would load actual image datasets like CelebA,
    COMPAS mugshots, or medical images pre-split by demographic groups.

    Parameters
    ----------
    n_samples_group_0 : int
        Number of samples for sensitive group 0
    n_samples_group_1 : int
        Number of samples for sensitive group 1
    image_shape : tuple
        Shape of synthetic images (height, width)
    bias_strength : float
        Amount of bias to introduce between groups

    Returns
    -------
    tuple
        (X_group_0, X_group_1, y_group_0, y_group_1)
    """

    # Generate synthetic "image" features (flattened)
    n_features = np.prod(image_shape)

    # Group 0: Generate base features
    X_group_0_flat, y_group_0 = make_classification(
        n_samples=n_samples_group_0,
        n_features=n_features,
        n_informative=min(20, n_features//2),
        n_redundant=min(10, n_features//4),
        n_classes=2,
        class_sep=1.0,
        random_state=42
    )

    # Group 1: Generate slightly different features (introduces bias)
    X_group_1_flat, y_group_1 = make_classification(
        n_samples=n_samples_group_1,
        n_features=n_features,
        n_informative=min(20, n_features//2),
        n_redundant=min(10, n_features//4),
        n_classes=2,
        class_sep=1.0 - bias_strength,  # Slightly harder classification
        random_state=43
    )

    # Add group-specific bias to features
    # Group 0 tends to have higher pixel values in top-left
    X_group_0_flat[:, :n_features//4] += bias_strength

    # Group 1 tends to have higher pixel values in bottom-right
    X_group_1_flat[:, -n_features//4:] += bias_strength

    # Reshape to "image" format (samples, height, width)
    X_group_0 = X_group_0_flat.reshape(-1, *image_shape)
    X_group_1 = X_group_1_flat.reshape(-1, *image_shape)

    # Normalize to [0, 1] range like real images
    X_group_0 = (X_group_0 - X_group_0.min()) / (X_group_0.max() - X_group_0.min())
    X_group_1 = (X_group_1 - X_group_1.min()) / (X_group_1.max() - X_group_1.min())

    return X_group_0, X_group_1, y_group_0, y_group_1


def train_calibrated_image_classifier(X, y, group_name):
    """
    Train a calibrated classifier for image data.

    Parameters
    ----------
    X : array-like of shape (n_samples, height, width)
        Image data
    y : array-like of shape (n_samples,)
        Labels
    group_name : str
        Name of the group for logging

    Returns
    -------
    CalibratedClassifierCV
        Trained calibrated classifier
    """
    print(f"  Training classifier for {group_name}...")
    print(f"    Data shape: {X.shape}, Labels: {np.bincount(y)}")

    # Create and train base classifier
    base_clf = SimpleImageClassifier(image_shape=X.shape[1:])

    # Split data for training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit base classifier
    base_clf.fit(X_train, y_train)

    # Calibrate the classifier
    calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    calibrated_clf.fit(X_val, y_val)

    # Test calibration quality
    val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_accuracy = (val_preds == y_val).mean()

    print(f"    Validation accuracy: {val_accuracy:.3f}")
    print(f"    Probability range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")

    return calibrated_clf


def visualize_sample_images(X_group_0, X_group_1, y_group_0, y_group_1):
    """Visualize sample images from each group."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Show samples from group 0
    for i in range(4):
        idx = np.where(y_group_0 == i % 2)[0][i // 2]
        axes[0, i].imshow(X_group_0[idx], cmap='gray')
        axes[0, i].set_title(f'Group 0, Label {y_group_0[idx]}')
        axes[0, i].axis('off')

    # Show samples from group 1
    for i in range(4):
        idx = np.where(y_group_1 == i % 2)[0][i // 2]
        axes[1, i].imshow(X_group_1[idx], cmap='gray')
        axes[1, i].set_title(f'Group 1, Label {y_group_1[idx]}')
        axes[1, i].axis('off')

    plt.suptitle('Sample Images from Each Sensitive Group')
    plt.tight_layout()
    plt.show()


def main():
    """Main example workflow for image fairness."""
    print("FairPareto Image Fairness Example")
    print("=" * 45)

    # 1. Create synthetic image data
    print("\n1. Creating synthetic image dataset...")

    X_group_0, X_group_1, y_group_0, y_group_1 = create_synthetic_image_data(
        n_samples_group_0=300,
        n_samples_group_1=200,
        image_shape=(16, 16),  # Small images for faster processing
        bias_strength=0.4
    )

    print(f"   Group 0: {X_group_0.shape} images, labels: {np.bincount(y_group_0)}")
    print(f"   Group 1: {X_group_1.shape} images, labels: {np.bincount(y_group_1)}")

    # Check for bias
    group_0_pos_rate = y_group_0.mean()
    group_1_pos_rate = y_group_1.mean()
    print(f"   Group 0 positive rate: {group_0_pos_rate:.3f}")
    print(f"   Group 1 positive rate: {group_1_pos_rate:.3f}")
    print(f"   Label bias: {abs(group_1_pos_rate - group_0_pos_rate):.3f}")

    # 2. Visualize sample images
    print("\n2. Visualizing sample images...")
    try:
        visualize_sample_images(X_group_0, X_group_1, y_group_0, y_group_1)
    except:
        print("   (Matplotlib display not available)")

    # 3. Train calibrated classifiers for each group
    print("\n3. Training calibrated classifiers for each group...")

    clf_group_0 = train_calibrated_image_classifier(X_group_0, y_group_0, "Group 0")
    clf_group_1 = train_calibrated_image_classifier(X_group_1, y_group_1, "Group 1")

    # 4. Compute fairness-performance Pareto front
    print("\n4. Computing fairness-performance Pareto front...")

    # Use custom gamma values for faster computation
    custom_gammas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    fair_clf = FairParetoClassifier(
        strategy='histogram',
        n_bins=6,  # Fewer bins for faster computation
        k=2,
        gamma_values=custom_gammas,
        verbose=True
    )

    # Use fit_presplit for pre-split image data
    fair_clf.fit_presplit(
        X_group_0, X_group_1,
        y_group_0, y_group_1,
        clf_group_0, clf_group_1
    )

    # 5. Analyze results
    print("\n5. Analyzing fairness-performance trade-offs...")

    pareto_front = fair_clf.get_pareto_front()
    valid_results = {gamma: (1-acc) for gamma, acc in pareto_front.items()
                    if not np.isnan(acc)}

    print(f"   Successfully computed {len(valid_results)} Pareto points")
    print("\n   Image Fairness-Performance Trade-offs:")
    print("   γ (Fairness Level) | Optimal Accuracy")
    print("   " + "-" * 35)

    for gamma in sorted(valid_results.keys()):
        accuracy = valid_results[gamma]
        print(f"   {gamma:8.2f}          | {accuracy:8.3f}")

    # 6. Visualization
    print("\n6. Plotting Pareto front...")

    try:
        plt.figure(figsize=(10, 6))

        gammas = sorted(valid_results.keys())
        accuracies = [valid_results[g] for g in gammas]

        plt.plot(gammas, accuracies, 'ro-', linewidth=2, markersize=8,
                label='Image Data Pareto Front')
        plt.xlabel('Fairness Level (γ) - Statistical Parity Tolerance', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Fairness-Performance Pareto Front for Image Classification\n' +
                 f'Groups: {len(y_group_0)} vs {len(y_group_1)} images', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add context annotations
        if len(gammas) >= 2:
            plt.annotate(f'Perfect Fairness\n(γ={gammas[0]})',
                        xy=(gammas[0], accuracies[0]),
                        xytext=(gammas[0] + 0.05, accuracies[0] - 0.02),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red')

            plt.annotate(f'No Fairness\n(γ={gammas[-1]})',
                        xy=(gammas[-1], accuracies[-1]),
                        xytext=(gammas[-1] - 0.1, accuracies[-1] + 0.02),
                        arrowprops=dict(arrowstyle='->', color='blue'),
                        fontsize=9, color='blue')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("   (Matplotlib not available for plotting)")

    # 7. Real-world usage recommendations
    print("\n7. Real-world usage recommendations:")
    print("   • For actual image data, use proper CNN architectures (ResNet, EfficientNet)")
    print("   • Consider using pre-trained models with transfer learning")
    print("   • Ensure proper image preprocessing (normalization, augmentation)")
    print("   • Use larger datasets for more reliable Pareto front estimation")
    print("   • Consider computational costs when choosing n_bins and gamma_values")

    print("\n8. Example dataset adaptations:")
    print("   • Medical images: Split by hospital, region, or demographic")
    print("   • Face recognition: Split by race, gender, or age groups")
    print("   • Satellite imagery: Split by geographic region or time period")
    print("   • X-ray classification: Split by scanner type or patient demographics")

    print("\nImage fairness example completed successfully!")


if __name__ == "__main__":
    main()