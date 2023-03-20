import csv
import matplotlib.pyplot as plt
from numpy import ndarray


class RegressionModel:
    parameter_precision: int = 0  # Number of decimal places to round parameters to
    experimental_data: tuple[list[float], ...] = None  # Data from CSV file
    n: int = 0  # Number of input variables/features
    m: int = 0  # Number of training examples
    X: ndarray = None  # Design matrix
    y: ndarray = None  # Labels vector
    beta: ndarray = None  # Weights vector

    def parse_input(
        self,
        input_file_path: str,
    ) -> tuple[list[float], ...]:
        """Parses a CSV file & returns a tuple with each entry containing the data as a list."""
        with open(input_file_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            experimental_data: tuple[list[float], ...] = tuple()
            # "Append" each row (converted to a list of floats) to the tuple
            for row in csv_reader:
                experimental_data += ([float(data) for data in row],)
        if len(experimental_data) < 2:
            raise ValueError("Input must have at least two lines.")
        return experimental_data

    def visualize(self) -> None:
        """Plots the experimental data along with the linear regression curve.
        If there are 4 or
        more input variables, the regression equation is printed instead.
        """
        plt.ion()
        if self.n == 1:
            self.plot_univariate()
        elif self.n == 2:
            self.plot_bivariate()
        elif self.n == 3:
            self.plot_trivariate()
        else:
            self.print_regression_equation()

    def __init__(
        self,
        input_file_path: str,
    ) -> None:
        self.experimental_data = self.parse_input(input_file_path)
        self.n = len(self.experimental_data) - 1  # Minus one to account for the labels
        self.m = len(self.experimental_data[0])
        self.X = self.construct_design_matrix()
        self.y = self.experimental_data[-1]
