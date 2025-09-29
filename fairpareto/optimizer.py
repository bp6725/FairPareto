"""
MIFPO optimization implementation using CVXPY.
"""

import numpy as np
import cvxpy as cvx
import dccp
import matplotlib.pyplot as plt


class EvelPertoFront:
    """
    Evaluates the Pareto front for fairness-performance trade-offs using MIFPO.

    This class implements the Model Independent Fairness-Performance Optimization
    (MIFPO) algorithm using convex optimization to compute optimal trade-offs.

    Parameters
    ----------
    n : int
        Number of probability bins for group 1.
    m : int
        Number of probability bins for group 0.
    k : int
        Number of representation points per bin pair.
    alpha : float
        Group size ratio: size_group_1 / (size_group_0 + size_group_1).
    pu : array-like of shape (n, 1)
        Probability centers for group 1.
    pv : array-like of shape (m, 1)
        Probability centers for group 0.
    U : array-like of shape (n, 1)
        Frequency weights for group 1 bins.
    V : array-like of shape (m, 1)
        Frequency weights for group 0 bins.
    loss : {'ACC', 'entropy'}, default='ACC'
        Loss function to optimize.
    """

    def __init__(self, n, m, k, alpha, pu, pv, U, V, loss="ACC", gamma_values = None):
        self.n = n
        self.m = m
        self.k = k
        self.alpha = alpha
        self.pu = pu
        self.pv = pv
        self.U = U
        self.V = V
        self.loss = loss
        self.results = {}
        self.gamma_values = gamma_values

    def solve_for_gamma(self, gamma):
        """
        Solve the MIFPO optimization problem for a specific fairness level.

        Parameters
        ----------
        gamma : float
            Fairness level (total variation distance bound).

        Returns
        -------
        tuple
            (Ru_to_v, Rv_to_u) - optimal transportation matrices.
        """
        # Define optimization variables
        Ru_to_v = {i: cvx.Variable((self.n, self.m)) for i in range(self.k)}
        Rv_to_u = {i: cvx.Variable((self.m, self.n)) for i in range(self.k)}
        R_chiu_to_v = {i: cvx.Variable((self.n, self.m)) for i in range(self.k)}
        R_chiv_to_u = {i: cvx.Variable((self.m, self.n)) for i in range(self.k)}

        # Define constraints
        constraints = []

        # Probability constraints
        constraints.append(sum(Ru_to_v.values()) @ np.ones(self.m) == np.ones(self.n))
        constraints.append(sum(Rv_to_u.values()) @ np.ones(self.n) == np.ones(self.m))
        constraints.append(cvx.sum(cvx.sum([v for v in R_chiu_to_v.values()])) == 1)
        constraints.append(cvx.sum(cvx.sum([v for v in R_chiv_to_u.values()])) == 1)

        # Non-negativity and fairness constraints
        for kk in range(self.k):
            constraints.append(cvx.min(Ru_to_v[kk]) >= 0)
            constraints.append(cvx.min(Rv_to_u[kk]) >= 0)
            constraints.append(cvx.min(R_chiu_to_v[kk]) >= 0)
            constraints.append(cvx.min(R_chiv_to_u[kk]) >= 0)

            # Fairness constraint (total variation bound)
            constraints.append(
                cvx.multiply(Ru_to_v[kk], self.U) + gamma * R_chiu_to_v[kk]
                == cvx.multiply(self.V, Rv_to_u[kk]).T + gamma * R_chiv_to_u[kk].T
            )

        # Define objective function
        cost = []
        for kk in range(self.k):
            if self.loss == "ACC":
                # Classification accuracy objective
                qv = (1 - self.alpha) * cvx.multiply(Rv_to_u[kk], self.V).T
                qv_pv = (1 - self.alpha) * cvx.multiply(
                    self.pv, cvx.multiply(Rv_to_u[kk], self.V)
                ).T
                qu = self.alpha * cvx.multiply(Ru_to_v[kk], self.U)
                qu_pu = self.alpha * cvx.multiply(
                    self.pu, cvx.multiply(Ru_to_v[kk], self.U)
                )

                cost.append(
                    cvx.sum(cvx.minimum(qv + qu - (qv_pv + qu_pu), qv_pv + qu_pu))
                )

            elif self.loss == "entropy":
                # Entropy-based objective
                a = (1 - self.alpha) * cvx.multiply(Rv_to_u[kk], self.V).T
                b = self.alpha * cvx.multiply(Ru_to_v[kk], self.U)

                a_pa = (1 - self.alpha) * cvx.multiply(
                    self.pv, cvx.multiply(Rv_to_u[kk], self.V)
                ).T
                b_pb = self.alpha * cvx.multiply(
                    self.pu, cvx.multiply(Ru_to_v[kk], self.U)
                )

                a_one_minus_pa = (1 - self.alpha) * cvx.multiply(
                    (1 - self.pv), cvx.multiply(Rv_to_u[kk], self.V)
                ).T
                b_one_minus_pb = self.alpha * cvx.multiply(
                    (1 - self.pu), cvx.multiply(Ru_to_v[kk], self.U)
                )

                _cost = cvx.sum(cvx.entr(a_pa + b_pb))
                cost.append(_cost)

        # Solve optimization problem
        prob = cvx.Problem(cvx.Minimize(cvx.sum(cost)), constraints)

        try:
            res = prob.solve(method="dccp")[0]
            if res is None :
                print(f"Warning: Optimization failed for gamma={gamma}, setting result to NaN")
                self.results[gamma] = np.nan
            else:
                self.results[gamma] = res
        except Exception as e:
            print(f"Warning: Optimization error for gamma={gamma}: {str(e)}")
            self.results[gamma] = np.nan

        return Ru_to_v, Rv_to_u

    def run(self, gamma_values):
        """
        Run optimization for multiple fairness levels.

        Parameters
        ----------
        gamma_values : list or float
            Fairness level(s) to optimize for.

        Returns
        -------
        dict or tuple
            If gamma_values is a list, returns None (results stored in self.results).
            If gamma_values is a single value, returns optimization variables.
        """
        if isinstance(gamma_values, list):
            for gamma in gamma_values:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"    - Solving for gamma = {gamma:.3f}")
                self.solve_for_gamma(gamma)
        else:
            return self.solve_for_gamma(gamma_values)

    def plot_results(self, additional_results=None):
        """
        Plot the computed Pareto front.

        Parameters
        ----------
        additional_results : dict, optional
            Additional results to plot for comparison.
        """
        if additional_results is not None:
            sorted_keys = sorted(additional_results.keys())
            sorted_values = [additional_results[key] for key in sorted_keys]
            plt.plot(sorted_keys, sorted_values, marker="o", color="b", label="Additional")

        plt.plot(
            list(self.results.keys()),
            list(self.results.values()),
            marker="o",
            linestyle="-",
            color="r",
            label="Pareto Front"
        )
        plt.xlabel("Fairness Level (γ)")
        plt.ylabel("Optimal Accuracy")
        plt.title("Fairness-Performance Pareto Front")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_results(self):
        """
        Get the computed Pareto front results.

        Returns
        -------
        dict
            Dictionary mapping fairness levels to optimal accuracy values.
        """
        return self.results