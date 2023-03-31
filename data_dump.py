import pymongo
import pandas as pd
import json
from insurance.config import mongo_client


# Provide the mongodb localhost url to connect python to mongodb.
#mongo_client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATA_FILE_PATH = '/config/workspace/insurance.csv'
DATABASE_NAME = "insurance_1"
COLLECTION_NAME = "insurance_prediction"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f'Rows and Columns: {df.shape}')

    #convert dataframe to json format so that we can dump the records into mongodb
    df.reset_index(drop=True,inplace=True)
    
    json_records  = list(json.loads(df.T.to_json()).values())
    print(json_records[0])
    
    #inserted converted json record to mongodb
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)