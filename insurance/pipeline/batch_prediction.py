import os,sys
import pandas as pd 
import numpy as np
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.utils import load_object
from insurance.predictor import ModelResolver
from datetime import datetime
PREDICTION_DIR = "prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading File : {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace = True)

        logging.info(f"Loading the transformer to transform dataset")
        transformer = load_object(file_path = model_resolver.get_latest_transformer_path())

        logging.info(f"Loading the tranformer encoder to encode the categorical independent variables")
        target_encoder = load_object(file_path = model_resolver.get_latest_target_path() )

        logging.info(f"Loading the model to predict ")
        model = load_object(file_path = model_resolver.get_latest_model_path())

        input_feature_names = list(transformer.feature_names_in_)

        logging.info(f'{input_feature_names}')
        for i in input_feature_names:       
            if df[i].dtypes =='object':
                df[i] =target_encoder.fit_transform(df[i])
        
        input_arr = transformer.transform(df[input_feature_names])

        prediction = model.predict(input_arr)

        df["prediction"]=prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise InsuranceException(e, sys)
