import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from src import utils
import argparse
import joblib
import json


def eval_metrics(actual, pred, pred_proba):
    """ Takes in the ground truth labels, predictions labels, and prediction probabilities.
        Returns the accuracy, f1, auc_roc, log_loss scores.
    """
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average="weighted")
    roc_auc = roc_auc_score(actual, pred_proba[:, 1])
    log_loss_score = log_loss(actual, pred_proba)
    return accuracy, f1, roc_auc, log_loss_score


def train_and_evaluate(config_path):
    config = utils.read_params(config_path)
    train_data_path = config["split_dataset"]["train_path"]
    dev_data_path = config["split_dataset"]["dev_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    C = config["estimators"]["LogisticRegression"]["params"]["C"]
    l1_ratio = config["estimators"]["LogisticRegression"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    dev = pd.read_csv(dev_data_path, sep=",")

    train_y = train[target]
    dev_y = dev[target]

    train_x = train.drop([f'{str(target[0])}', 'id', 'question1',
                         'question2', 'question1_tokens', 'question2_tokens'], axis=1)
    dev_x = dev.drop([f'{str(target[0])}', 'id', 'question1',
                     'question2', 'question1_tokens', 'question2_tokens'], axis=1)

    lr = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver="liblinear",
        random_state=random_state)

    lr.fit(train_x, train_y)

    pred_proba = lr.predict_proba(dev_x)
    pred = lr.predict(dev_x)

    (accuracy, f1, roc_auc, log_loss_score) = eval_metrics(dev_y, pred, pred_proba)

    print("Elasticnet model (C=%f, l1_ratio=%f):" % (C, l1_ratio))
    print("  ACCURACY: %s" % accuracy)
    print("  F1: %s" % f1)
    print("  ROC AUC: %s" % roc_auc)
    print("  LOG LOSS: %s" % log_loss_score)

#####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy_score": accuracy,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
            "log_loss": log_loss_score
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "C": C,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)
#####################################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
