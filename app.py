import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('tree_regressor_model.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)                           ## values will be in json- dictionary format, so we need to use .values() to fetch data
    new_data=[list(data.values())]        ## converting to 2-D list or dataframe
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])            ## predict should be same as in Html page form action
def predict():

    data=[float(x) for x in request.form.values()]                  ## converting to list and float value in order of features  ## from.values will fetch all input data from html form
    final_features = [np.array(data)]                                ## converting to 2-D list or dataframe
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))



if __name__=="__main__":
    app.run(debug=True)


