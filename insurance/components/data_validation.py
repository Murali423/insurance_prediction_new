import os,sys
from insurance.entity import config_entity,artifact_entity
import numpy as np
import pandas as pd 
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance import utils
from scipy.stats import ks_2samp
from typing import Optional
from insurance.config import TARGET_COLUMN


class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig,
                   data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f'{">>"*20}Data Validation {"<<"*20}')
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise InsuranceException(error_message = e, error_detail = sys)

    def drop_missing_value_columns(self, df:pd.DataFrame ,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop the missing value columns which are greater than specified threshold
        
        df: Accept the dataframe
        report_key_name: missing key value which will be added into report
        ==========================================================================================
        returns Pandas dataframe atleast single column is available
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isnull().sum()/df.shape[0]
            logging.info(f'dropping the missing columns which are greaterthan {threshold}')
            drop_columns = null_report[null_report>threshold].index
            logging.info(f"Columns to be dropped:{drop_columns}")
            self.validation_error[report_key_name] = list(drop_columns)
            df.drop(list(drop_columns),inplace=True,axis = 1)
            # return None no columns left
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)


    def is_required_column_exist(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        """
        This function retuns the is the required column exist or not 

        base_df : first dataframe without missing columns
        current_df : Data frame after missing columns
        report_key_name: missing key value which will be added into report
        ===================================================================
        returns True or False
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column  in base_columns:
                if base_column not in current_columns:
                    logging.info(f'{base_column} not available')
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True

        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        """
        This function retuns the is the required column exist or not 

        base_df : first dataframe without missing columns
        current_df : Data frame after missing columns
        report_key_name: missing key value which will be added into report
        ===================================================================
        returns True or False
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            drift_report = dict()

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                logging.info(f"Hypothesis for {base_column}:{base_data.dtype} and {current_data.dtype}")
                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue > 0.05:
                    logging.info(f"we have accepted the null hypothesis")
                    drift_report[base_column] = {
                        "pvalue":float(same_distribution.pvalue),
                        "same_distrubition":True
                    }
                else:
                    logging.info(f"We accepted alternate hypothesis and data drift has occured")
                    drift_repot[base_column] = {
                        "pvalue" : float(same_distribution.pvalue),
                        "same_distrubution":False
                    }
            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise InsuranceException(error_message=e, error_detail= sys)

    def initiate_data_validataion(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading Base DataFrame")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"Drop null values colums from base df")
            base_df.replace({"na":np.NAN},inplace=True)
            base_df = self.drop_missing_value_columns(df = base_df, report_key_name = "missing_values_with_base_df" )

            logging.info(f"Reading the Train DataFrame")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Dropping the missing columns")
            train_df.replace({"na":np.NAN},inplace = True)
            train_df = self.drop_missing_value_columns(df = train_df , report_key_name ="missing_values_with_train_df")

            logging.info(f'Reading the Test dataframe')
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info("Dropping the missing columns")
            test_df.replace({"na":np.NAN},inplace = True)
            test_df = self.drop_missing_value_columns(df = test_df, report_key_name = "missing_values_with_test_df")

            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convet_columns_float(df = base_df, exclude_columns = exclude_columns)
            train_df = utils.convet_columns_float(df = train_df, exclude_columns = exclude_columns)
            test_df = utils.convet_columns_float(df = test_df, exclude_columns = exclude_columns)

            logging.info(f"Is all required columns present in train df")
            train_df_column_status = self.is_required_column_exist(base_df = base_df, current_df = train_df, report_key_name = "missing_columns_within_train_dataset")
            logging.info(f"Is all required columns present in train df")
            test_df_column_status = self.is_required_column_exist(base_df = base_df, current_df = test_df, report_key_name = "missing_columns_within_train_dataset")

            if train_df_column_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
                self.data_drift(base_df = base_df, current_df = train_df, report_key_name = "data_drift_within_train_dataset")
            if test_df_column_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
                self.data_drift(base_df = base_df, current_df = test_df, report_key_name = "data_drift_within_test_dataset")
            
            #write report
            logging.info('Write report in yaml file')
            utils.write_yaml_file(file_path = self.data_validation_config.report_file_path, data = self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path = self.data_validation_config.report_file_path)
            logging.info(f'Data validation artifact: {data_validation_artifact}')
            return data_validation_artifact


        except Exception as e:
            raise InsuranceException(e, sys)
    