import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "df8M3vzdyoURu0po9wsFSEpklo1DUZCke1aG5Kh2xHRu"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open('svm.pkl','rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": [["gre","toefel","univ","cgpa"]], "values": features}]}
    
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/327c7505-d5bd-4d66-9760-70994d2e8fea/predictions?version=2022-11-26', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    res=response_scoring.json()['predictions'][0]['values'][0][0]
    
    #prediction = model.predict(features)
    if res==1:
        return render_template("goto.html")
    else:
        return render_template("rejected.html")

if __name__ == "__main__":
    flask_app.run(debug=True)