# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from src import utils
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocess_text_column


def split_dataset(config_path):
    """ Runs scripts to split the features data from (../into) into
        train, dev and test datasets ready to be fed into models (saved in ../processed).
    """
    config = utils.read_params(config_path)
    feature_data_path = config["build_features"]["feature_dataset_path"]
    train_data_path = config["split_dataset"]["train_path"]
    dev_data_path = config["split_dataset"]["dev_path"]
    test_data_path = config["split_dataset"]["test_path"]
    dev_size = config["split_dataset"]["dev_size"]
    test_size = config["split_dataset"]["test_size"]
    random_state = config["base"]["random_state"]

    df = utils.get_data(feature_data_path)

    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state)
    train, dev = train_test_split(
        train, test_size=dev_size, random_state=random_state)

    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    dev.to_csv(dev_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    split_dataset(config_path=parsed_args.config)
