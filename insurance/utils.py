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
        logging.info(f'Row and columns in df: {df.shape}')
        if '_id' in df.columns:
            logging.info(f"Dropping column: _id ")
            df.drop(['_id'],axis=1,inplace = True)
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
    """
    This fucnion  will convrt the columns to float
    df : Dataframe to be converted
    exclude_columns : List of collumns to be excluded
    ========================================================
    returns the Dataframe 
    """
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtype != 'O':
                    df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise InsuranceException(error_message = e, error_detail = sys)

def save_object(file_path:str, obj:object) ->None:
    """
    This fucnion  will save the object using dill
    file_path : path of file where yaml can store
    object : object to be stored 
    ========================================================
    returns None
    """
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path),exist_ok = True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Exited from the save_object method of utils")
    except Exception as e:
        raise InsuranceException(e, sys)

def load_object(file_path:str)->object:
    """
    This fucnion  will load the object using dill
    file_path : path of file where yaml can store
    ========================================================
    returns object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys)

def save_numpy_array_data(file_path:str,array:np.array)->None:
    """
    This fucnion  will save the array with numpy
    file_path : path of file where yaml can store
    array : Array to be stored in numpy
    ========================================================
    returns numpy object
    """
    try:
        logging.info("Entered the save_numpy method of utils")
        os.makedirs(os.path.dirname(file_path),exist_ok = True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
        logging.info("Exited from the save_numpy method of utils")
    except Exception as e:
        raise InsuranceException(e, sys)

def load_numpy_array_data(file_path:str)->np.array:
    """
    This fucnion  will load the numpy array using numpy
    file_path : path of file where yaml can store
    ========================================================
    returns numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys)

def adj_r2score(score,X,y):
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