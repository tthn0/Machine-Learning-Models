import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .regression_model import RegressionModel


class NormEqReg(RegressionModel):
    def compute_beta(
        self,
        precision: int,
    ) -> ndarray:
        # Transpose design matrix & multiply by design matrix
        XTX: ndarray = self.X.T @ self.X
        # Transpose design matrix & multiply by y
        XTy: ndarray = self.X.T @ self.y
        # Multiply inverse of XTX by XTy
        beta: ndarray = np.linalg.pinv(XTX) @ XTy
        # Return beta with all values rounded to the specified precision
        return np.array([round(beta_i, precision) for beta_i in beta])

    def rotate_plot(
        self,
        ax: plt.axes,
    ) -> None:
        angle = 0
        while True:
            # Normalize the angle to the range [-180, 180]
            angle_norm = (angle + 180) % 360 - 180
            # Update the axis view
            ax.view_init(elev=20, azim=angle_norm)
            plt.draw()
            plt.pause(0.01)
            angle += 2

    def __init__(
        self,
        input_file_path: str,
        parameter_precision: int,
    ) -> None:
        super().__init__(input_file_path)
        self.parameter_precision: int = parameter_precision
        self.beta = self.compute_beta(parameter_precision)
        self.visualize()
