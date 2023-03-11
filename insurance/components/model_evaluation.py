import os,sys
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity import config_entity,artifact_entity
from insurance import utils
import numpy as np
import pandas as pd
from typing import Optional
from insurance.config import TARGET_COLUMN
from insurance.predictor import ModelResolver
from sklearn.metrics import r2_score


class ModelEvaluation:

    def __init__(self,model_evaluation_config: config_entity.ModelEvaluationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f'{">>"*20} Model Evaluation {"<<"*20}')
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder
            logging.info(f"If saved model has model it will compare\
                          which model is best trained or model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True, 
                                                        improved_accuracy = None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            logging.info("Finding location of transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_path()

            logging.info("Previous trained objects of transformer, model and target encoder")
            transformer = utils.load_object(file_path = transformer_path)
            model = utils.load_object(file_path = model_path)
            target_encoder = utils.load_object(file_path = target_encoder_path)

            logging.info(f"Loading current trained model objects")
            current_transformer = utils.load_object(file_path = self.data_transformation_artifact.transform_object_path)
            current_model = utils.load_object(file_path = self.model_trainer_artifact.model_path)
            current_target = utils.load_object(file_path = self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"test dataframe : {test_df.shape}")
            target_df = test_df[TARGET_COLUMN]
            y_true = target_df

            input_feature_name = list(transformer.feature_names_in_)
            for i in input_feature_name:
                if test_df[i].dtype == 'O':
                    test_df[i] = target_encoder.fit_transform(test_df[i])
            
            input_arr = transformer.transform(test_df[input_feature_name])
            y_prd = model.predict(input_arr)
            print(f"Prediction using previous model: {y_prd[:5]}")
            previous_model_r2_score = r2_score(y_true=y_true, y_pred=y_prd)
            logging.info(f"r2_score: {previous_model_r2_score} ")
            logging.info(f"test dataframe : {test_df.shape}")
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_path)
            previous_model_adj_r2_score = utils.adj_r2score(score = previous_model_r2_score , X = test_arr[:,:-1], y = test_arr[:,-1])
            logging.info(f"Accuracy using previous trained model: {previous_model_adj_r2_score}")

            logging.info("Calculating accuracy using current model")
            input_current_feature_name = list(current_transformer.feature_names_in_)
            input_current_arr = current_transformer.transform(test_df[input_current_feature_name])
            y_current_prd = model.predict(input_current_arr)
            print(f"Prediction using trained model: {y_current_prd[:5]}")
            current_model_r2_score = r2_score(y_true=y_true, y_pred=y_current_prd)
            current_model_adj_r2_score = utils.adj_r2score(score= current_model_r2_score , X = test_arr[:,:-1], y = test_arr[:,-1])
            if current_model_adj_r2_score < previous_model_adj_r2_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")
            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True, 
                                            improved_accuracy = current_model_adj_r2_score -previous_model_adj_r2_score )
            logging.info(f"Model eval artifact: {model_evaluation_artifact}")                                      
            return model_evaluation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)          