import sys
from typing import Optional

from sklearn.metrics import f1_score

from src.components.data_drift_detection import DataDriftDetector
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from src.data_access.proj1_data import Proj1Data
from src.entity.artifact_entity import ModelMonitoringArtifact
from src.entity.config_entity import ModelMonitoringConfig, DataIngestionConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.cloud_storage.aws_storage import SimpleStorageService
from src.pipline.training_pipeline import TrainPipeline
from src.utils.main_utils import read_yaml_file
from src.utils.feature_engineering import apply_manual_feature_engineering


class ModelMonitor:
    """
    Runs a periodic check of data drift (reference vs. current MongoDB pull) and
    of the current production model's F1 score, deciding whether the training
    pipeline needs to be re-run automatically.
    """

    def __init__(self, monitoring_config: ModelMonitoringConfig = ModelMonitoringConfig(),
                 data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.monitoring_config = monitoring_config
            self.data_ingestion_config = data_ingestion_config
            self.s3 = SimpleStorageService()
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def _fetch_current_data(self):
        try:
            proj1_data = Proj1Data()
            return proj1_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def _fetch_reference_data(self):
        try:
            reference_available = self.s3.s3_key_path_available(
                bucket_name=self.monitoring_config.bucket_name,
                s3_key=self.monitoring_config.s3_reference_data_key)
            if not reference_available:
                logging.info("No reference data snapshot found in S3 yet; skipping drift check.")
                return None
            return self.s3.read_csv(
                filename=self.monitoring_config.s3_reference_data_key,
                bucket_name=self.monitoring_config.bucket_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def _evaluate_current_model_f1(self, current_df) -> Optional[float]:
        try:
            estimator = Proj1Estimator(
                bucket_name=self.monitoring_config.bucket_name,
                model_path=self.monitoring_config.s3_model_key_path)
            if not estimator.is_model_present(model_path=self.monitoring_config.s3_model_key_path):
                logging.info("No production model found in S3 yet; skipping F1 check.")
                return None

            x = current_df.drop(columns=[TARGET_COLUMN], axis=1)
            y = current_df[TARGET_COLUMN]
            x = apply_manual_feature_engineering(x, self._schema_config)

            y_pred = estimator.predict(x)
            return float(f1_score(y, y_pred))
        except Exception as e:
            raise MyException(e, sys) from e

    def run_monitoring(self) -> ModelMonitoringArtifact:
        """
        Method Name :   run_monitoring
        Description :   Pulls the latest data, checks it for drift against the
                        reference snapshot and checks the production model's F1
                        score, then triggers retraining if either signal fires.

        Output      :   Returns a ModelMonitoringArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Starting model monitoring run.")

            current_df = self._fetch_current_data()
            reference_df = self._fetch_reference_data()

            data_drift_artifact = None
            if reference_df is not None:
                data_drift_artifact = DataDriftDetector(self.monitoring_config).detect_drift(
                    reference_df=reference_df, current_df=current_df)

            current_f1_score = self._evaluate_current_model_f1(current_df)

            drift_detected = data_drift_artifact.drift_detected if data_drift_artifact else False
            f1_below_threshold = current_f1_score is not None and current_f1_score < self.monitoring_config.f1_threshold
            retraining_triggered = drift_detected or f1_below_threshold

            if drift_detected and f1_below_threshold:
                reason = f"Data drift detected and F1 score ({current_f1_score}) fell below threshold ({self.monitoring_config.f1_threshold})."
            elif drift_detected:
                reason = "Data drift detected in the current data pull."
            elif f1_below_threshold:
                reason = f"F1 score ({current_f1_score}) fell below threshold ({self.monitoring_config.f1_threshold})."
            else:
                reason = "No drift detected and F1 score is within the acceptable threshold."

            logging.info(f"Monitoring decision: retraining_triggered={retraining_triggered}, reason={reason}")

            if retraining_triggered:
                logging.info("Triggering automatic retraining via TrainPipeline.")
                TrainPipeline().run_pipeline()

            return ModelMonitoringArtifact(
                data_drift_artifact=data_drift_artifact,
                current_f1_score=current_f1_score,
                retraining_triggered=retraining_triggered,
                reason=reason,
            )
        except Exception as e:
            raise MyException(e, sys) from e
