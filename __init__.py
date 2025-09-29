"""
FairPareto: A package for computing fairness-performance Pareto fronts using MIFPO.

This package implements the Model Independent Fairness-Performance Optimization (MIFPO)
algorithm for computing optimal trade-offs between fairness and performance in binary
classification tasks.
"""

from fairpareto.core import FairParetoClassifier

__version__ = "0.1.0"
__author__ = "Binyamin Perets and Mark Kozdoba"
__email__ = ""

__all__ = ["FairParetoClassifier"]