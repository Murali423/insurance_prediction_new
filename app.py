from flask import Flask,redirect,url_for,render_template,request
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.predictor import ModelResolver
import pandas as pd 
import numpy as np 


app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/success/<int:score>')
def success(score):
    exp = score
    logging.info(f"Predicted insurance is : {score}")
    return render_template('result.html',result=exp)


### Result checker submit html page
@app.route('/submit',methods=['POST','GET'])
def submit():
    insu = {}
    if request.method=='POST':
        insu['age'] = int(request.form['age'])
        insu['gender'] = request.form['gender']
        insu['bmi'] = float(request.form['bmi'])
        insu['childern'] = int(request.form['Childern'])
        insu['smoker'] = request.form['Smoker']
        insu['region'] = request.form['Region']
        logging.info(f"Creating dataframe from dictionary")
        df = pd.DataFrame(insu, index=[0])
        df.replace({"na":np.NAN},inplace = True)

        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
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
    logging.info(f"insurance form has submitted")
    return redirect(url_for('success',score=prediction))

    



if __name__=='__main__':
    app.run(debug=True)