import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Response

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('Home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]          # getting values in list format
    final_features = [np.array(data)]                 # covertng to 2 d array
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('Home.html', prediction_text = "Airfoil pressure is {}".format(output))




if __name__ == "__main__":
    app.run(debug=True)
