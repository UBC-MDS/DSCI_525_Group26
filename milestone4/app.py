from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load your model here
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(X_predict):    
    return model.predict(X_predict)

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 5 climate model outputs.
    """



# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():

    content = request.json  # this extracts the JSON content we sent

    #X_predict = pd.DataFrame(content)
        
    data = np.array(content["data"]).reshape(1, -1)  # this extracts the data attribute from the JSON content
    data = data.reshape(1, -1)
    
    #data = pd.DataFrame(data)
    
    prediction = return_prediction(data)
    #print(prediction)

    results =  {'predictions': prediction.tolist()}

    #for col in data:
    #    results['input_' + col] = data[col].values.tolist()

    return jsonify(results)
