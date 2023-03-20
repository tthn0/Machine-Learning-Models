import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .norm_eq_reg import NormEqReg


class NormEqQuadReg(NormEqReg):
    def construct_design_matrix(self) -> ndarray:
        """Constructs the design matrix from the experimental data.
        Size of the design matrix: (m)x(2n+1). (2n columns for each input variable and +1 column for bias term.)
        """
        # Initialize design matrix w/ all ones
        design_matrix: ndarray = np.ones((self.m, 2 * self.n + 1))
        # Fill columns of design matrix
        for i in range(self.n):
            design_matrix[:, 2 * i + 1] = np.array(self.experimental_data[i])
            design_matrix[:, 2 * i + 2] = np.array(self.experimental_data[i]) ** 2
        return design_matrix

    def plot_univariate(self) -> None:
        # Plot experimental data
        plt.scatter(*self.experimental_data, color="red")
        # Define quadratic regression curve
        x1 = np.linspace(
            np.amin(self.experimental_data[0]),
            np.amax(self.experimental_data[0]),
        )
        y = self.beta[0] + self.beta[1] * x1 + self.beta[2] * x1**2
        # plot quadratic regression curve
        plt.plot(x1, y, color="green")
        # Add labels & show plot
        plt.title(f"y = %s + %sx + %sx²" % tuple(self.beta), fontsize=15)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.show()

    def plot_bivariate(self) -> None:
        ax = plt.axes(projection="3d")
        # Plot experimental data
        ax.scatter(*self.experimental_data, color="red")
        # Define regression surface
        x1 = np.linspace(
            np.amin(self.experimental_data[0]),
            np.amax(self.experimental_data[0]),
        )
        x2 = np.linspace(
            np.amin(self.experimental_data[1]),
            np.amax(self.experimental_data[1]),
        )
        x1, x2 = np.meshgrid(x1, x2)
        y = (
            self.beta[0]
            + self.beta[1] * x1
            + self.beta[2] * x1**2
            + self.beta[3] * x2
            + self.beta[4] * x2**2
        )
        # Plot regression surface
        ax.plot_surface(x1, x2, y, color="green")
        # Add labels
        plt.title(
            "y = %s + %sx₁ + %sx₁² + %sx₂ + %sx₂²" % tuple(self.beta),
            fontsize=15,
        )
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("y", fontsize=20)
        self.rotate_plot(ax)

    def plot_trivariate(self) -> None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # Plot experimental data
        scatter_plot = ax.scatter(
            *self.experimental_data,
            c=self.y,
            cmap="rainbow",
        )
        # Add a legend (colorbar)
        fig.colorbar(scatter_plot, ax=ax)
        # Add labels
        plt.title(
            "y = %s + %sx₁ + %sx₁² + %sx₂ + %sx₂² + %sx₃ + %sx₃²" % tuple(self.beta),
            fontsize=15,
        )
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("x₃", fontsize=20)
        self.rotate_plot(ax)

    def print_regression_equation(self) -> None:
        for i, beta_i in enumerate(self.beta):
            if i == 0:
                print(f"y = {beta_i} + ", end="")
            elif i % 2 == 1:
                print(f"{beta_i}(x_{i//2+1}) + ", end="")
            elif i % 2 == 0 and i != len(self.beta) - 1:
                print(f"{beta_i}(x_{i//2})² + ", end="")
            elif i == len(self.beta) - 1:
                print(f"{beta_i}(x_{i//2})²")
