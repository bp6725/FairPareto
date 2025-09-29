# FairPareto: Optimal Fairness-Performance Trade-offs

[![PyPI version](https://badge.fury.io/py/fairpareto.svg)](https://badge.fury.io/py/fairpareto)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

FairPareto is a Python package that computes the optimal fairness-performance Pareto front for binary classification using the **Model Independent Fairness-Performance Optimization (MIFPO)** algorithm.

## Key Features

- 🎯 **Optimal Trade-offs**: Computes the theoretical optimal Pareto front between fairness and performance
- 🔧 **sklearn Compatible**: Follows scikit-learn's API conventions for seamless integration  
- 📊 **Multiple Data Types**: Supports both tabular and non-tabular data (images, text, etc.)
- ⚡ **Automatic Calibration**: Optional built-in calibrated XGBoost classifier with hyperparameter tuning
- 📈 **Visualization**: Built-in plotting for Pareto front visualization
- 🔍 **Model Independent**: Provides theoretical benchmarks independent of specific ML models

## Installation

```bash
pip install fairpareto
```

## Quick Start

### Tabular Data

```python
from fairpareto import FairParetoClassifier
import pandas as pd

# Load your data (X should include the sensitive attribute column)
clf = FairParetoClassifier(sensitive_column='gender')
clf.fit(X, y)

# Get the optimal Pareto front
pareto_front = clf.pareto_front_
print(pareto_front)
# {0.0: 0.85, 0.1: 0.82, 0.2: 0.78, ...}  # {fairness_level: optimal_accuracy}

# Plot the results
clf.plot_pareto_front()
```

### Pre-split Data (Non-tabular)

For images, text, or other non-tabular data where you need custom classifiers:

```python
# Your pre-trained calibrated classifiers for each sensitive group
clf_group_0 = your_trained_classifier_0  # Must have predict_proba()
clf_group_1 = your_trained_classifier_1  # Must have predict_proba()

clf = FairParetoClassifier()
clf.fit_presplit(X_0, X_1, y_0, y_1, clf_group_0, clf_group_1)

pareto_front = clf.get_pareto_front()
```

## Parameters

### FairParetoClassifier

- **`sensitive_column`** *(str)*: Name of the sensitive attribute column in X (required for tabular data)
- **`strategy`** *(str, default='histogram')*: Binning strategy - 'histogram', 'eqz_bins', or 'exact'
- **`n_bins`** *(int, default=10)*: Number of bins for probability discretization
- **`k`** *(int, default=2)*: MIFPO optimization parameter (higher = more complex representations)
- **`verbose`** *(bool, default=False)*: Print progress information

## Understanding the Output

The Pareto front is returned as a dictionary mapping **fairness levels (γ)** to **optimal accuracy**:

- **γ = 0.0**: Perfect fairness (equal statistical parity), potentially lower accuracy
- **γ = 1.0**: No fairness constraints, maximum possible accuracy  
- **Intermediate values**: Optimal trade-offs between fairness and performance

```python
# Example output
{
    0.0: 0.75,    # Perfect fairness: 75% accuracy
    0.1: 0.82,    # Small fairness violation: 82% accuracy  
    0.2: 0.85,    # Moderate fairness violation: 85% accuracy
    ...
    1.0: 0.90     # No fairness constraints: 90% accuracy
}
```

## Advanced Usage

### Custom Calibrated Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from fairpareto.calibration import CustomCalibratedClassifier

# Use your own base classifier
base_clf = RandomForestClassifier()
calibrated_clf = CustomCalibratedClassifier(base_estimator=base_clf)

clf = FairParetoClassifier(sensitive_column='race')
clf.fit(X, y, calibrated_classifier=calibrated_clf)
```

### Different Binning Strategies

```python
# Equal frequency binning (better for skewed probability distributions)
clf = FairParetoClassifier(
    sensitive_column='age_group',
    strategy='eqz_bins',
    n_bins=15
)

# Exact probabilities (no binning, slower but more precise)
clf = FairParetoClassifier(
    sensitive_column='gender', 
    strategy='exact'
)
```

## Mathematical Background

FairPareto implements the MIFPO algorithm from the paper "Efficient Fairness-Performance Pareto Front Computation", which:

1. **Factorizes representations** through probability distributions over outcomes
2. **Uses invertibility properties** to reduce the optimization space  
3. **Solves convex-concave optimization** problems to find optimal trade-offs
4. **Provides theoretical guarantees** that the computed front is optimal

The algorithm measures:
- **Fairness**: Total variation distance between group distributions (statistical parity)
- **Performance**: Classification accuracy under optimal prediction

## Requirements

- Python 3.7+
- numpy >= 1.19.0
- pandas >= 1.3.0  
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- cvxpy >= 1.2.0
- aequitas >= 0.42.0
- matplotlib >= 3.3.0

## Examples

### Complete Example with Real Data

```python
import pandas as pd
from sklearn.datasets import make_classification
from fairpareto import FairParetoClassifier

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
sensitive = np.random.binomial(1, 0.4, size=len(y))

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['sensitive_attr'] = sensitive

# Fit FairPareto classifier
clf = FairParetoClassifier(
    sensitive_column='sensitive_attr',
    strategy='histogram',
    n_bins=10,
    verbose=True
)

clf.fit(df, y)

# Analyze results
pareto_front = clf.get_pareto_front()
print("Fairness-Performance Trade-offs:")
for gamma, accuracy in sorted(pareto_front.items()):
    print(f"γ={gamma:.2f}: {accuracy:.3f} accuracy")

# Visualize
clf.plot_pareto_front()
```

## API Reference

### FairParetoClassifier Methods

#### `fit(X, y, calibrated_classifier=None)`
Fit the classifier using tabular data.

**Parameters:**
- `X`: DataFrame with features including sensitive attribute
- `y`: Binary target labels (0/1)
- `calibrated_classifier`: Optional pre-trained calibrated classifier

#### `fit_presplit(X_0, X_1, y_0, y_1, clf_0, clf_1)`
Fit using pre-split data and classifiers.

**Parameters:**
- `X_0, X_1`: Feature arrays for each sensitive group
- `y_0, y_1`: Labels for each sensitive group  
- `clf_0, clf_1`: Calibrated classifiers for each group

#### `get_pareto_front()`
Returns the computed Pareto front as a dictionary.

#### `plot_pareto_front(title=None, ax=None)`
Plot the Pareto front curve.

## Troubleshooting

### Common Issues

**CVXPY Solver Failures:**
```python
# Some gamma values may fail to optimize
pareto_front = clf.get_pareto_front()
valid_points = {g: acc for g, acc in pareto_front.items() if not np.isnan(acc)}
```

**Memory Issues with Large Datasets:**
```python
# Reduce the number of bins or k parameter
clf = FairParetoClassifier(
    sensitive_column='attr',
    n_bins=5,    # Reduce from default 10
    k=2          # Keep default
)
```

**Slow Performance:**
```python
# Use histogram strategy (fastest)
clf = FairParetoClassifier(
    sensitive_column='attr',
    strategy='histogram',  # Fastest option
    n_bins=8               # Fewer bins = faster
)
```

### Development Setup

```bash
git clone https://github.com/your-username/fairpareto.git
cd fairpareto
pip install -e ".[dev]"
pytest tests/
```

## Citation

If you use FairPareto in your research, please cite:

```bibtex
@misc{kozdoba2024efficientfairnessperformanceparetocomputation,
      title={Efficient Fairness-Performance Pareto Front Computation}, 
      author={Mark Kozdoba and Binyamin Perets and Shie Mannor},
      year={2024},
      eprint={2409.17643},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2409.17643}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
