from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity import  config_entity
import sys,os
from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.components.data_transformation import DataTransformation
from insurance.components.model_trainer import ModelTrainer
from insurance.components.model_evaluation import ModelEvaluation
from insurance.components.model_pusher import ModelPusher
from insurance.pipeline.training_pipeline import start_training_pipeline
from insurance.pipeline.batch_prediction import start_batch_prediction

file_path = "/config/workspace/insurance.csv"


if __name__ == "__main__":
     try:
          # output_file = start_training_pipeline()
          # print(output_file)
          #Batch Prediction
          output_file = start_batch_prediction(input_file_path = file_path)
          print(output_file)
     except Exception as e:
          raise InsuranceException( e, sys)