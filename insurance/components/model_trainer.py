import os,sys
from insurance.entity import config_entity,artifact_entity
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance import utils
from typing import Optional
import pandas as pd
import numpy as np 
from insurance.config import TARGET_COLUMN




class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                      data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise InsuranceException(error_message = e, error_detail = sys)