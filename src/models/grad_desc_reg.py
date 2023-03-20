import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .regression_model import RegressionModel


class GradDescReg(RegressionModel):
    alpha: float = 10e-4  # Learning rate
    cost: float = (
        np.inf
    )  # Cost of hypothesis with the given weights at previous iteration

    def j(self) -> float:
        """Returns the cost (MSE) of the hypothesis with the given weights."""
        m: int = len(self.y)
        return np.sum((self.f() - self.y) ** 2) / (2 * m)

    def f(self) -> ndarray:
        """Returns a matrix of hypotheses for each row in the design matrix."""
        hypothesis: ndarray = self.X @ self.beta
        if np.any(np.isinf(hypothesis)) or np.any(np.isnan(hypothesis).any()):
            raise ValueError(
                "Failed to converge. Try making learning rate (alpha) smaller."
            )
        return hypothesis

    def compute_gradients(self) -> ndarray:
        """Returns a vector of gradients for each weight in the weights vector."""
        y_hats: ndarray = self.f()
        residuals: ndarray = y_hats - self.y
        # Initialize a vector of gradients
        num_weights: int = len(self.beta)
        gradients: ndarray = np.zeros(num_weights)
        # Compute gradient for each weight using partial derivatives
        for n in range(num_weights):
            x_n: ndarray = self.X[:, n]
            gradients[n] = np.sum(residuals @ x_n) / self.m
        return gradients

    def update_weights(self) -> None:
        self.beta = np.round(
            self.beta - self.alpha * self.compute_gradients(),
            self.parameter_precision,
        )

    def __init__(
        self,
        input_file_path: str,
        parameter_precision: int,
    ) -> None:
        super().__init__(input_file_path)
        self.parameter_precision: int = parameter_precision
        # Initialize weights vector, one weight for each column in design matrix
        self.beta: ndarray = np.zeros(len(self.X.T))
        self.visualize()
