import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
import joblib

params_path = "params.yaml"

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir = config["model_dir"]
    model = joblib.load(model_dir)
    prediction = model.predict(data).tolist()[0]
    return prediction

def api_response(request):
    data = np.array([list(request.values())])
    response = predict(data)
    response = {"response": response}
    return response