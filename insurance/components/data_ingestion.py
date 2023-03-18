import os,sys
import pandas as pd
import numpy as np 
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance import utils
from insurance.entity import config_entity,artifact_entity
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                 collection_name = self.data_ingestion_config.collection_name)

            logging.info(f"Save data in feature store having the shape: {df.shape} ")
            logging.info(f"Columns for the dataframe are : {df.columns} ")

            #Replace with Nan
            df.replace(to_replace="na",value=np.NAN,inplace=True)

            #Save data in feature store
            logging.info('Create feature store folder if not available')
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            #training and test splitting.
            logging.info('Spliting the data into train and test dataset')
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size, random_state = 1)

            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            train_path_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(name = train_path_dir, exist_ok= True)

            logging.info('creating the dataset directory folder for test if not available')
            test_path_dir = os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(name =test_path_dir,exist_ok=True )

            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index= False, header =True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index = False, header = True)

            logging.info("Preparing the artifact")
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path, 
                test_file_path = self.data_ingestion_config.test_file_path)

            logging.info(f'Data Ingestion artifact : {data_ingestion_artifact} ')
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(e, sys)