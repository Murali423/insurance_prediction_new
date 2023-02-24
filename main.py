from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
import sys,os



if __name__ == "__main__":
     try:
          training_pipeline_config = TrainingPipelineConfig()
          data_ingestion_config = DataIngestionConfig(training_pipeline_config = training_pipeline_config)
          print(data_ingestion_config.to_dict())
     except Exception as e:
          print(e)