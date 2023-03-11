import os,sys
from insurance.entity import config_entity,artifact_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import pandas as pd
import numpy as np
from insurance import utils
from insurance.predictor import ModelResolver

class ModelPusher:

    def __init__(self, model_pusher_config: config_entity.ModelPusherConfig,
                       model_trainer_artifact: artifact_entity.ModelTrainerArtifact,
                       data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_resolver = ModelResolver(model_registry= self.model_pusher_config.save_model_dir)
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_pusher(self,)->artifact_entity.ModelPusherArtifact:
        try:
            logging.info("load the transformer model and target encoder objects")
            transformer = utils.load_object(file_path = self.data_transformation_artifact.transform_object_path)
            model = utils.load_object(file_path = self.model_trainer_artifact.model_path)
            target_encoder = utils.load_object(file_path = self.data_transformation_artifact.target_encoder_path)

            #saving the objects to model pusher dir
            logging.info(f"Saving the objects into model pusher directory")
            utils.save_object(file_path = self.model_pusher_config.pusher_transformer_path, obj = transformer)
            utils.save_object(file_path = self.model_pusher_config.pusher_model_path, obj = model)
            utils.save_object(file_path = self.model_pusher_config.pusher_target_encoder_path, obj = target_encoder)

            #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            utils.save_object(file_path= transformer_path, obj = transformer)
            utils.save_object(file_path = model_path, obj = model)
            utils.save_object(file_path = target_encoder_path, obj = target_encoder)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(pusher_model_dir = self.model_pusher_config.pusher_model_dir, 
                                   saved_model_dir = self.model_pusher_config.save_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e, sys)