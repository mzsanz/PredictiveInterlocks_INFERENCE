from flask import Flask, request
import os
from app.src.models.predict import predict_pipeline
import warnings

# Remove unneeded warnings
warnings.filterwarnings('ignore')

# -*- coding: utf-8 -*-
app = Flask(__name__)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))


# using @app.route to manage routers (GET method)
@app.route('/', methods=['GET'])
def root():
    """
        Function to manage the output from root path.

        Returns:
           dict.  Output message
    """
    return {'Project':'Predictive Interlocks'}


# path to start the inference pipeline (POST method)
@app.route('/predict', methods=['POST'])
def predict_route():
    """
        Function to start the inference pipeline.

        Returns:
           dict.  Output message (prediction)
    """
     
    # Obtain data from the request
    data = request.get_json()

    print(data)

    # Start the execution of the inference pipeline
    y_pred = predict_pipeline(data)

    return {'Predicted BP': y_pred}


# main
if __name__ == '__main__':
    # ejecución de la app
    app.run(host='0.0.0.0', port=port, debug=True)
