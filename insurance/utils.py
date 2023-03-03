import os,sys
import pandas as pd
import numpy as np 
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.config import mongo_client
import yaml
import dill

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    ====================================
    Params:
    database_name: name of database
    collection_name : name of the collection
    ====================================
    returns the collection data as pandas Dataframe
    """
    try:
        logging.info(f'Reading data from database: {database_name} and collection: {collection_name}')
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f'Found columns:{df.columns}')
        if '_id' in df.columns:
            logging.info(f"Dropping column: _id ")
            df.drop("_id",axis=1)
        logging.info(f'Row and columns in df: {df.shape}')
        return df
    except Exception as e:
        raise InsuranceException(e, sys)

def write_yaml_file(file_path:str,data:dict):
    """
    This fucnion  will write the dict to yaml files
    file_path : path of file where yaml can store
    data : dictionary files which need to kept in yaml file 
    ========================================================
    returns the yaml file 
    """
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(name = file_dir, exist_ok = True)
        with open(file_path,'w') as file:
            yaml.dump(data,file)
    except Exception as e:
        raise InsuranceException( e, sys)

def convet_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtype != 'O':
                    df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise InsuranceException(error_message = e, error_detail = sys)