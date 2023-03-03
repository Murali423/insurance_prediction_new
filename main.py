from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity import  config_entity
import sys,os
from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation



if __name__ == "__main__":
     try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config = training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config = data_ingestion_config)
          print(data_ingestion.initiate_data_ingestion())
          data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
          #validation Code
          data_validation_config = config_entity.DataValidationConfig(training_pipeline_config =training_pipeline_config )
          data_validation = DataValidation(data_validation_config = data_validation_config , 
                              data_ingestion_artifact = data_ingestion_artifact)
          data_validation_artifact = data_validation.initiate_data_validataion()
     except Exception as e:
          raise InsuranceException( e, sys)