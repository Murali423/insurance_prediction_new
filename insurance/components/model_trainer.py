import os,sys
from insurance.entity import config_entity,artifact_entity
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance import utils
from typing import Optional
import pandas as pd
import numpy as np 
from insurance.config import TARGET_COLUMN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                      data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_report = dict()
        except Exception as e:
            raise InsuranceException(error_message = e, error_detail = sys)

    def fine_tune(self,model,param):
        '''
        This function used to hyperparameter the given model for best parameter to get good accuracy.
        model: Model for which we need to find the best param
        param: On which prams we need to make hyper parameter tuenning
        cv: Cross Validation socre
        '''
        try:
            #code for RandomizedSearch CV
            logging.info(f"Entered into hyperparameter tuening")
            logging.info(f'Given are {model}:{param}')
            hyper_model = RandomizedSearchCV(model,param,cv = 3)
            return hyper_model
        except Exception as e:
            raise InsuranceException(e, sys)

    def train_model(self,model,x,y):
        '''
        This function Used to fit the model to given dataset
        X : Independent features of Dataset
        y : Dependente features of Dataset
        ========================================================
        Returns: Model which is trained on dataset
        '''
        try:
            logging.info(f'Enterd to training of model')
            trainer = model
            trainer.fit(x,y)
            return model
        except Exception as e:
            raise InsuranceException(e, sys)

    def performance_metrics(self,y_real , y_pred):
        '''
        This function measure the mean absolute error, mean square error, r2 score
        y_real : original values of y
        y_pred : predicted values of y
        ========================================================
        Returns: MSE, MAE, R2_score
        '''
        MSE = mean_squared_error(y_real,y_pred)
        MAE = mean_absolute_error(y_real,y_pred)
        RMSE = np.sqrt(MSE)
        r2 =  r2_score(y_real,y_pred)
        return MSE,MAE,RMSE,r2

    def adj_r2score(self, score,X,y):
        '''
        This function measure the adjusted r2 value
        score : r2_score of the model
        X : Independente features
        y: Dependente feature
        ================================================
        Returns: adj_r2 value of the model
        '''
        adjR = 1 - ( 1-score ) * ( len(y) - 1 ) / ( len(y) - X.shape[1] - 1 )
        return adjR

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            report = dict()
            logging.info(f'Loading train and test array.')
            train_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            gb_ens = GradientBoostingRegressor(random_state=0)
            model = self.train_model(model=gb_ens ,x=x_train,y=y_train)

            logging.info(f'Finding the performance metrics for train dataset')
            y_pred_train = model.predict(x_train)
            mse_train,mae_train,rmse_train,r2_score_train = self.performance_metrics(y_real = y_train, y_pred = y_pred_train)
            adj_r2_train = self.adj_r2score(score = r2_score_train , X = x_train, y = y_train )
            report['train'] = {
                "training_mse" : float(mse_train),
                "training_mae": float(mae_train),
                "training_rmse": float(rmse_train),
                "training_r2_score" : float(r2_score_train),
                "training_adj_r2_score": float(adj_r2_train)
            }

            logging.info(f'Finding the performance metrics for test dataset')
            y_pred_test = model.predict(x_test)
            mse_test, mae_test, rmse_test, r2_score_test = self.performance_metrics(y_real = y_test, y_pred = y_pred_test)
            adj_r2_test = self.adj_r2score(score = r2_score_test , X = x_test, y = y_test )
            report['test'] = {
                "test_mse" : float(mse_test),
                "test_mae": float(mae_test),
                "test_rmse": float(rmse_test),
                "test_r2_score" : float(r2_score_test),
                "test_adj_r2_score": float(adj_r2_test)
            }

            param = {
                "learning_rate":[0.01,0.1],
                "n_estimators" : [150,200]
            }

            GB_hyper = self.fine_tune(model = gb_ens, param = param)
            hyper_model = self.train_model(model = GB_hyper, x = x_train, y = y_train)
            logging.info(f'After hyper parameter tuening best parameters : {hyper_model.best_params_} and best score: {hyper_model.best_score_}')
            
            logging.info(f'Fining the performance metrics for Hyper parameter train dataset')
            hyper_y_pred_train = hyper_model.predict(x_train)
            mse_h_tr, mae_h_tr, rmse_h_tr, r2_score_h_tr = self.performance_metrics(y_real = y_train, y_pred = hyper_y_pred_train)
            adj_r2_h_tr = self.adj_r2score(score = r2_score_h_tr , X = x_train, y = y_train )
            report['hyper_train'] = {
                "test_mse" : float(mse_h_tr),
                "test_mae": float(mae_h_tr),
                "test_rmse": float(rmse_h_tr),
                "test_r2_score" : float(r2_score_h_tr),
                "test_adj_r2_score": float(adj_r2_h_tr)
            }
            
            logging.info(f'Fining the performance metrics for Hyper parameter test dataset')
            hyper_y_pred_test = hyper_model.predict(x_test)
            mse_h_test, mae_h_test, rmse_h_test, r2_score_h_test = self.performance_metrics(y_real = y_test, y_pred = hyper_y_pred_test)
            adj_r2_h_test = self.adj_r2score(score = r2_score_h_test , X = x_test, y = y_test )
            report['hyper_test'] = {
                "test_mse" : float(mse_h_test),
                "test_mae": float(mae_h_test),
                "test_rmse": float(rmse_h_test),
                "test_r2_score" : float(r2_score_h_test),
                "test_adj_r2_score": float(adj_r2_h_test)
            }

            logging.info(f"Checking if our model is underfitting or not")
            if adj_r2_h_test<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {adj_r2_h_test}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(adj_r2_h_tr - adj_r2_h_test)
            logging.info(f"difference of the model train and test: {diff}")

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=hyper_model)

            self.model_report = report
            #write report
            logging.info('Write report in yaml file')
            utils.write_yaml_file(file_path = self.model_trainer_config.model_trainer_report, data = self.model_report)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path = self.model_trainer_config.model_path,
                       r2_train_score = r2_score_h_tr, 
                       r2_test_score = r2_score_h_test, 
                       adj_r2_train_score = adj_r2_h_tr, 
                       adj_r2_test_score = adj_r2_h_test)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys)