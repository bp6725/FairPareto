"""
Basic usage example for FairPareto package.

This example demonstrates how to use FairPareto to compute optimal
fairness-performance trade-offs on synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from fairpareto.core import FairParetoClassifier


def create_synthetic_data(n_samples=500, n_features=8, bias_strength=0.3):
    """
    Create synthetic biased dataset for fairness evaluation.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    bias_strength : float
        Strength of bias to introduce (0 = no bias, 1 = strong bias)
    """
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        n_informative=int(0.8 * n_features),
        n_redundant=int(0.2 * n_features),
        n_clusters_per_class=1,
        random_state=42
    )

    # Create sensitive attribute with some correlation to target
    # This introduces bias in the dataset
    sensitive_prob = 0.3 + bias_strength * (y * 0.4)  # Base 30%, up to 70% for y=1
    sensitive = np.random.binomial(1, sensitive_prob, size=len(y))

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['gender'] = sensitive  # Our sensitive attribute

    return df, y


def main():
    """Main example workflow."""
    print("FairPareto Basic Usage Example")
    print("=" * 40)

    # 1. Generate synthetic biased data
    print("\n1. Generating synthetic biased dataset...")
    X, y = create_synthetic_data(n_samples=400, n_features=6, bias_strength=0.4)

    print(f"   Dataset shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)}")
    print(f"   Sensitive attribute distribution: {np.bincount(X['gender'])}")

    # Check for bias in the data
    group_0_positive_rate = y[X['gender'] == 0].mean()
    group_1_positive_rate = y[X['gender'] == 1].mean()
    print(f"   Positive rate in group 0: {group_0_positive_rate:.3f}")
    print(f"   Positive rate in group 1: {group_1_positive_rate:.3f}")
    print(f"   Bias (difference): {abs(group_1_positive_rate - group_0_positive_rate):.3f}")

    # 2. Compute optimal Pareto front
    print("\n2. Computing optimal fairness-performance Pareto front...")

    clf = FairParetoClassifier(
        sensitive_column='gender',
        strategy='histogram',
        n_bins=8,  # Fewer bins for faster example
        k=2,
        # gamma_values = [0,0.5,1],
        verbose=True
    )

    # Fit on data
    clf.fit(X, y)

    # 3. Analyze results
    print("\n3. Analyzing results...")
    pareto_front = clf.get_pareto_front()

    # Filter out any NaN values from failed optimizations
    valid_results = {gamma: acc for gamma, acc in pareto_front.items()
                     if not np.isnan(acc)}

    print(f"   Successfully computed {len(valid_results)} Pareto points")
    print("\n   Fairness-Performance Trade-offs:")
    print("   γ (Fairness Level) | Optimal Accuracy")
    print("   " + "-" * 35)

    for gamma in sorted(valid_results.keys()):
        accuracy = valid_results[gamma]
        print(f"   {gamma:8.2f}          | {accuracy:8.3f}")

    # 4. Interpretation
    print("\n4. Interpretation:")
    if len(valid_results) >= 2:
        min_gamma = min(valid_results.keys())
        max_gamma = max(valid_results.keys())

        fair_accuracy = valid_results[min_gamma]
        unfair_accuracy = valid_results[max_gamma]

    # 5. Visualization
    print("\n5. Creating visualization...")

    try:
        plt.figure(figsize=(10, 6))

        # Plot Pareto front
        gammas = sorted(valid_results.keys())
        accuracies = [1-valid_results[g] for g in gammas]

        plt.plot(gammas, accuracies, 'bo-', linewidth=2, markersize=8, label='Optimal Pareto Front')
        plt.xlabel('Fairness Level (γ) - Statistical Parity Tolerance', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Fairness-Performance Pareto Front\n(Lower γ = More Fair)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add annotations for key points
        if len(gammas) >= 2:
            plt.annotate('Perfect Fairness',
                         xy=(gammas[0], accuracies[0]),
                         xytext=(gammas[0] , accuracies[0] ),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=10, color='red')

            plt.annotate('No Fairness Constraints',
                         xy=(gammas[-1], accuracies[-1]),
                         xytext=(gammas[-1] - 0.15, accuracies[-1] + 0.02),
                         arrowprops=dict(arrowstyle='->', color='blue'),
                         fontsize=10, color='blue')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("   (Matplotlib not available for plotting)")

    print("\n6. Summary:")
    print("   • FairPareto computed the theoretical optimal trade-offs")
    print("   • These results serve as benchmarks for fair ML algorithms")
    print("   • Any practical fair classifier should aim to approach this front")
    print("   • The package provides a model-independent evaluation tool")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()