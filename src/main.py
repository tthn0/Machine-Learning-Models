import json

from models.grad_desc_lin_reg import GradDescLinReg
from models.grad_desc_quad_reg import GradDescQuadReg

from models.norm_eq_lin_reg import NormEqLinReg
from models.norm_eq_quad_reg import NormEqQuadReg


def parse_config(
    config_file_path: str,
) -> dict[str, int | str]:
    """Returns the JSON data as a dictionary."""
    with open(config_file_path) as config_file:
        return json.load(config_file)


def main():
    config: dict[str, int | str] = parse_config("config.json")
    regression_method: str = config["regression_method"]
    regression_type: str = config["regression_type"]
    input_file_path: str = config["input_file_path"]
    parameter_precision: int = config["parameter_precision"]

    try:
        if regression_method == "gradient" and regression_type == "linear":
            GradDescLinReg(input_file_path, parameter_precision)
        elif regression_method == "gradient" and regression_type == "quadratic":
            GradDescQuadReg(input_file_path, parameter_precision)
        elif regression_method == "normal" and regression_type == "linear":
            NormEqLinReg(input_file_path, parameter_precision)
        elif regression_method == "normal" and regression_type == "quadratic":
            NormEqQuadReg(input_file_path, parameter_precision)
        else:
            error_message: str = """Invalid configuration file.
            'regression_method' must be either 'gradient' or 'normal'.
            'regression_type' must be either 'linear' or 'quadratic'.
            """
            raise ValueError(error_message)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
