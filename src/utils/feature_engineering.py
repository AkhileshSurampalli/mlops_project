import sys

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


def _map_gender_column(df: DataFrame) -> DataFrame:
    """Map Gender column to 0 for Female and 1 for Male."""
    logging.info("Mapping 'Gender' column to binary values")
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
    return df


def _drop_id_column(df: DataFrame, schema_config: dict) -> DataFrame:
    """Drop the schema-configured id/drop column if it exists."""
    logging.info("Dropping id column")
    drop_col = schema_config['drop_columns']
    if drop_col in df.columns:
        df = df.drop(drop_col, axis=1)
    return df


def _create_dummy_columns(df: DataFrame) -> DataFrame:
    """Create dummy variables for categorical features."""
    logging.info("Creating dummy variables for categorical features")
    df = pd.get_dummies(df, drop_first=True)
    return df


def _rename_columns(df: DataFrame) -> DataFrame:
    """Rename specific columns and ensure integer types for dummy columns."""
    logging.info("Renaming specific columns and casting to int")
    df = df.rename(columns={
        "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
        "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
    })
    for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
        if col in df.columns:
            df[col] = df[col].astype('int')
    return df


def apply_manual_feature_engineering(df: DataFrame, schema_config: dict) -> DataFrame:
    """
    Applies the manual feature engineering steps (gender mapping, id column drop,
    dummy variable creation, column renaming) shared by data transformation,
    model evaluation, and model monitoring.
    """
    try:
        df = _map_gender_column(df)
        df = _drop_id_column(df, schema_config)
        df = _create_dummy_columns(df)
        df = _rename_columns(df)
        return df
    except Exception as e:
        raise MyException(e, sys) from e
