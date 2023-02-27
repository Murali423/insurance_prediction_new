import os,sys
import pandas as pd
import numpy as np 
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.config import mongo_client

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

