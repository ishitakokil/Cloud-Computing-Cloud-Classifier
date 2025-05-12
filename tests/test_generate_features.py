import pytest
import pandas as pd
import numpy as np
from src.generate_features import generate_features

# ---------- Normalization Tests ----------

def test_norm_range_happy():
    config = {
        "calculate_norm_range": {
            "IR_norm_range": {
                "min_col": "IR_min",
                "max_col": "IR_max",
                "mean_col": "IR_mean"
            }
        }
    }
    df = pd.DataFrame({"IR_min": [1], "IR_max": [5], "IR_mean": [2]})
    result = generate_features(df, config)
    assert "IR_norm_range" in result.columns
    assert np.isclose(result["IR_norm_range"].iloc[0], 2.0)

def test_norm_range_missing_column():
    config = {
        "calculate_norm_range": {
            "IR_norm_range": {
                "min_col": "IR_min",
                "max_col": "IR_max",
                "mean_col": "IR_mean"
            }
        }
    }
    df = pd.DataFrame({"IR_max": [5], "IR_mean": [2]})  # Missing IR_min
    with pytest.raises(ValueError):
        generate_features(df, config)

# ---------- Log Transform Tests ----------

def test_log_transform_happy():
    config = {
        "log_transform": {
            "log_entropy": "visible_entropy"
        }
    }
    df = pd.DataFrame({"visible_entropy": [0.5]})
    result = generate_features(df, config)
    assert "log_entropy" in result.columns
    assert np.isclose(result["log_entropy"].iloc[0], np.log(0.5 + 1e-5))

def test_log_transform_missing_column():
    config = {
        "log_transform": {
            "log_entropy": "visible_entropy"
        }
    }
    df = pd.DataFrame({})  # Missing visible_entropy
    with pytest.raises(ValueError):
        generate_features(df, config)

# ---------- Multiplication Tests ----------

def test_multiplication_happy():
    config = {
        "multiply": {
            "entropy_x_contrast": {
                "col_a": "visible_contrast",
                "col_b": "visible_entropy"
            }
        }
    }
    df = pd.DataFrame({
        "visible_entropy": [0.5],
        "visible_contrast": [2.0]
    })
    result = generate_features(df, config)
    assert "entropy_x_contrast" in result.columns
    assert np.isclose(result["entropy_x_contrast"].iloc[0], 1.0)

def test_multiplication_missing_column():
    config = {
        "multiply": {
            "entropy_x_contrast": {
                "col_a": "visible_contrast",
                "col_b": "visible_entropy"
            }
        }
    }
    df = pd.DataFrame({
        "visible_entropy": [0.5]
    })  # Missing visible_contrast
    with pytest.raises(ValueError):
        generate_features(df, config)


