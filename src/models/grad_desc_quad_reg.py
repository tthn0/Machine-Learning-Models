from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from time import sleep


from .grad_desc_reg import GradDescReg


class GradDescQuadReg(GradDescReg):
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

    def plot_univariate(
        self,
    ) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Plot experimental data
        ax.scatter(*self.experimental_data, color="red")

        # Set axis labels
        plt.xlabel("x₁", fontsize=20)
        plt.ylabel("y", fontsize=20)

        # Define quadratic regression curve
        x1 = np.linspace(
            np.amin(self.experimental_data[0]),
            np.amax(self.experimental_data[0]),
        )
        y = self.beta[0] + self.beta[1] * x1 + self.beta[2] * x1**2
        (regression_line,) = ax.plot(x1, y, color="green")

        # Train model for at most 100,000 iterations
        for i in range(100_000):
            self.update_weights()

            if abs(self.cost - self.j()) > 0.1 ** (self.parameter_precision):
                # If there is a significant change in the cost
                self.cost = self.j()
            else:
                try:
                    sleep(10e6)  # Keep plot on screen until user closes it
                except KeyboardInterrupt:
                    break

            if i % 10 == 0:
                # Update title & plot
                plt.title(f"y = %s + %sx + %sx²" % tuple(self.beta), fontsize=15)
                regression_line.set_ydata(
                    self.beta[0] + self.beta[1] * x1 + self.beta[2] * x1**2
                )
                fig.canvas.draw()
                fig.canvas.flush_events()

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
        # Add labels
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("y", fontsize=20)

        # Train model and update plot
        angle = 0
        converged = False
        while True:
            if not converged:
                self.update_weights()

                if abs(self.cost - self.j()) > 0.1 ** (self.parameter_precision):
                    # If there is a significant change in the cost
                    self.cost = self.j()
                else:
                    converged = True

                if angle % 2 == 0:
                    # Update title & surface
                    plt.title(
                        "y = %s + %sx₁ + %sx₁² + %sx₂ + %sx₂²" % tuple(self.beta),
                    )

                    y = (
                        self.beta[0]
                        + self.beta[1] * x1
                        + self.beta[2] * x1**2
                        + self.beta[3] * x2
                        + self.beta[4] * x2**2
                    )
                    if angle != 0:
                        surface.remove()
                    surface = ax.plot_surface(x1, x2, y, color="green")

            # Normalize the angle to the range [-180, 180]
            angle_norm = (angle + 180) % 360 - 180
            # Update the axis view
            ax.view_init(elev=20, azim=angle_norm)
            plt.draw()
            plt.pause(10e-10)
            angle += 2

    def plot_trivariate(
        self,
    ) -> None:
        fig = plt.figure()

        residuals: ndarray = abs(self.f() - self.y)
        max_residual = np.amax(residuals)

        # Define colormap
        cmap = LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

        ax = plt.axes(projection="3d")
        # Plot experimental data
        scatter_plot = ax.scatter(
            *self.experimental_data,
            c=residuals,
            cmap=cmap,
        )

        # Add a legend (colorbar)
        fig.colorbar(scatter_plot, ax=ax)

        # Add labels
        ax.set_xlabel("x₁", fontsize=20)
        ax.set_ylabel("x₂", fontsize=20)
        ax.set_zlabel("y", fontsize=20)

        # Train model and update plot
        angle = 0
        converged = False
        while True:
            if not converged:
                self.update_weights()

                if abs(self.cost - self.j()) > 0.1 ** (self.parameter_precision):
                    # If there is a significant change in the cost
                    self.cost = self.j()
                else:
                    converged = True

                if angle % 10 == 0:
                    # Update title & replot data
                    plt.title(
                        "y = %s + %sx₁ + %sx₁² + %sx₂ + %sx₂² + %sx₃ + %sx₃²"
                        % tuple(self.beta),
                        fontsize=15,
                    )
                    scatter_plot.remove()

                    residuals: ndarray = np.abs(self.f() - self.y)
                    # Set the residual to the max residual to make the colorbar more readable
                    residuals[-1] = max_residual

                    scatter_plot = ax.scatter(
                        *self.experimental_data,
                        c=residuals,
                        cmap=cmap,
                    )

            # Normalize the angle to the range [-180, 180]
            angle_norm = (angle + 180) % 360 - 180
            # Update the axis view
            ax.view_init(elev=20, azim=angle_norm)
            plt.draw()
            plt.pause(10e-10)
            angle += 1

    def print_regression_equation(self) -> None:
        # Train model for at most 100_000 iterations
        for _ in range(100_000):
            self.update_weights()

            if abs(self.cost - self.j()) > 0.1 ** (self.parameter_precision):
                # If there is a significant change in the cost
                self.cost = self.j()
                for i, beta_i in enumerate(self.beta):
                    if i == 0:
                        print(f"y = {beta_i} + ", end="")
                    elif i % 2 == 1:
                        print(f"{beta_i}(x_{i//2+1}) + ", end="")
                    elif i % 2 == 0 and i != len(self.beta) - 1:
                        print(f"{beta_i}(x_{i//2})² + ", end="")
                    elif i == len(self.beta) - 1:
                        print(f"{beta_i}(x_{i//2})²")
            else:
                break
