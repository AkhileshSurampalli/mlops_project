import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import ks_2samp, chi2_contingency

from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from src.entity.artifact_entity import DataDriftArtifact
from src.entity.config_entity import ModelMonitoringConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file


class DataDriftDetector:
    """
    Detects feature drift between a reference dataset (the data the current
    production model was trained on) and a current dataset (the latest pull),
    using the Kolmogorov-Smirnov test for numerical features and the
    Chi-square test for categorical features.
    """

    def __init__(self, monitoring_config: ModelMonitoringConfig):
        try:
            self.monitoring_config = monitoring_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def _numerical_drift_test(self, reference_series: pd.Series, current_series: pd.Series) -> float:
        statistic, p_value = ks_2samp(reference_series.dropna(), current_series.dropna())
        return float(p_value)

    def _categorical_drift_test(self, reference_series: pd.Series, current_series: pd.Series) -> float:
        categories = sorted(set(reference_series.dropna().unique()) | set(current_series.dropna().unique()))
        if len(categories) < 2:
            return 1.0

        reference_counts = reference_series.value_counts().reindex(categories, fill_value=0)
        current_counts = current_series.value_counts().reindex(categories, fill_value=0)
        contingency_table = np.array([reference_counts.values, current_counts.values])

        _, p_value, _, _ = chi2_contingency(contingency_table)
        return float(p_value)

    def detect_drift(self, reference_df: DataFrame, current_df: DataFrame) -> DataDriftArtifact:
        """
        Method Name :   detect_drift
        Description :   Compares reference vs. current feature distributions and
                        decides whether dataset-level drift has occurred.

        Output      :   Returns a DataDriftArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            numerical_columns = [col for col in self._schema_config["numerical_columns"] if col != TARGET_COLUMN]
            categorical_columns = self._schema_config["categorical_columns"]

            drift_report = {}
            drifted_features = []

            for column in numerical_columns:
                if column not in reference_df.columns or column not in current_df.columns:
                    continue
                p_value = self._numerical_drift_test(reference_df[column], current_df[column])
                is_drifted = p_value < self.monitoring_config.ks_pvalue_threshold
                drift_report[column] = {"test": "KS", "p_value": p_value, "drift_detected": is_drifted}
                if is_drifted:
                    drifted_features.append(column)

            for column in categorical_columns:
                if column not in reference_df.columns or column not in current_df.columns:
                    continue
                p_value = self._categorical_drift_test(reference_df[column], current_df[column])
                is_drifted = p_value < self.monitoring_config.ks_pvalue_threshold
                drift_report[column] = {"test": "Chi-square", "p_value": p_value, "drift_detected": is_drifted}
                if is_drifted:
                    drifted_features.append(column)

            total_features_checked = len(drift_report)
            drift_proportion = (len(drifted_features) / total_features_checked) if total_features_checked else 0.0
            dataset_drift_detected = drift_proportion >= self.monitoring_config.drift_proportion_threshold

            drift_report["summary"] = {
                "drifted_feature_count": len(drifted_features),
                "total_feature_count": total_features_checked,
                "drift_proportion": drift_proportion,
                "dataset_drift_detected": dataset_drift_detected,
            }

            write_yaml_file(file_path=self.monitoring_config.drift_report_file_path, content=drift_report, replace=True)
            logging.info(f"Data drift report: {drift_report}")

            return DataDriftArtifact(
                drift_detected=dataset_drift_detected,
                drifted_features=drifted_features,
                drift_report_file_path=self.monitoring_config.drift_report_file_path,
            )
        except Exception as e:
            raise MyException(e, sys) from e
