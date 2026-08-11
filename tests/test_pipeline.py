"""Unit tests for the preprocessing and feature engineering modules."""

import pandas as pd
import pytest

from src.preprocessing import Preprocessor
from src.features import FeatureEngineer


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "tenure": [1, 24, 60],
        "MonthlyCharges": [70.5, 55.0, 90.0],
        "TotalCharges": ["70.5", " ", "5400.0"],  # blank simulates real Telco data quirk
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "Churn": ["Yes", "No", "No"],
    })


class TestPreprocessor:
    def test_clean_handles_blank_total_charges(self, sample_raw_df):
        pre = Preprocessor()
        cleaned = pre.clean(sample_raw_df)
        assert cleaned["TotalCharges"].isna().sum() == 0

    def test_clean_drops_id_column(self, sample_raw_df):
        pre = Preprocessor()
        cleaned = pre.clean(sample_raw_df)
        assert "customerID" not in cleaned.columns

    def test_encode_produces_numeric_target(self, sample_raw_df):
        pre = Preprocessor()
        cleaned = pre.clean(sample_raw_df)
        encoded = pre.encode(cleaned, fit=True)
        assert set(encoded["Churn"].unique()).issubset({0, 1})

    def test_encode_categoricals_are_numeric(self, sample_raw_df):
        pre = Preprocessor()
        cleaned = pre.clean(sample_raw_df)
        encoded = pre.encode(cleaned, fit=True)
        assert pd.api.types.is_numeric_dtype(encoded["Contract"])


class TestFeatureEngineer:
    def test_adds_tenure_years(self, sample_raw_df):
        pre = Preprocessor()
        encoded = pre.encode(pre.clean(sample_raw_df), fit=True)
        fe = FeatureEngineer()
        transformed = fe.transform(encoded)
        assert "tenure_years" in transformed.columns
        assert transformed["tenure_years"].iloc[0] == pytest.approx(1 / 12)

    def test_avg_monthly_spend_no_divide_by_zero(self):
        df = pd.DataFrame({"tenure": [0], "TotalCharges": [50.0], "MonthlyCharges": [50.0]})
        fe = FeatureEngineer()
        result = fe.transform(df)
        assert result["avg_monthly_spend"].iloc[0] == 50.0  # divides by 1, not 0

    def test_is_new_customer_flag(self, sample_raw_df):
        pre = Preprocessor()
        encoded = pre.encode(pre.clean(sample_raw_df), fit=True)
        fe = FeatureEngineer()
        transformed = fe.transform(encoded)
        assert transformed["is_new_customer"].iloc[0] == 1  # tenure=1
        assert transformed["is_new_customer"].iloc[2] == 0  # tenure=60
