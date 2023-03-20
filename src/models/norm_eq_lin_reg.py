import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .norm_eq_reg import NormEqReg


class NormEqLinReg(NormEqReg):
    def construct_design_matrix(self) -> ndarray:
        """Constructs the design matrix from the experimental data.
        Size of the design matrix: (m)x(n+1). (n columns for each input variable and +1 column for bias term.)
        """
        # Initialize design matrix w/ all ones
        design_matrix: ndarray = np.ones((self.m, self.n + 1))
        # Fill columns of design matrix
        for i in range(self.n):
            design_matrix[:, i + 1] = np.array(self.experimental_data[i])
        return design_matrix

    def plot_univariate(
        self,
    ) -> None:
        # Plot experimental data
        plt.scatter(*self.experimental_data, color="red")
        # Define linear regression line
        x1 = np.linspace(
            np.amin(self.experimental_data[0]),
            np.amax(self.experimental_data[0]),
        )
        y = self.beta[0] + self.beta[1] * x1
        # plot quadratic regression curve
        plt.plot(x1, y, color="green")
        # Add labels & show plot
        plt.title(f"y = %s + %sx₁" % tuple(self.beta), fontsize=15)
        plt.xlabel("x₁", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.show()

    def plot_bivariate(
        self,
    ) -> None:
        ax = plt.axes(projection="3d")
        # Plot experimental data
        ax.scatter(*self.experimental_data, color="red")
        # Define regression plane
        x1 = np.linspace(
            np.amin(self.experimental_data[0]),
            np.amax(self.experimental_data[0]),
        )
        x2 = np.linspace(
            np.amin(self.experimental_data[1]),
            np.amax(self.experimental_data[1]),
        )
        x1, x2 = np.meshgrid(x1, x2)
        y = self.beta[0] + self.beta[1] * x1 + self.beta[2] * x2
        # Plot regression plane
        ax.plot_surface(x1, x2, y, color="green")
        # Add labels
        plt.title("y = %s + %sx₁ + %sx₂" % tuple(self.beta), fontsize=15)
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("y", fontsize=20)
        self.rotate_plot(ax)

    def plot_trivariate(
        self,
    ) -> None:
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
        plt.title("y = %s + %sx₁ + %sx₂ + %sx₃" % tuple(self.beta), fontsize=15)
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("x₃", fontsize=20)
        self.rotate_plot(ax)

    def print_regression_equation(self) -> None:
        for i, beta_i in enumerate(self.beta):
            if i == 0:
                print(f"y = {beta_i} + ", end="")
            elif i != len(self.beta) - 1:
                print(f"{beta_i}(x_{i}) + ", end="")
            else:
                print(f"{beta_i}(x_{i})")
