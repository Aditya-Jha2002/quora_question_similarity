# -*- coding: utf-8 -*-
import argparse
from src import utils
import shutil


def load_dataset(config_path):
    """ Script to get the data from intial data folder (../initial) into
        raw data folder for further processing (saved in ../raw).
    """
    config = utils.read_params(config_path)
    initial_data_path = config["data_source"]["labeled_source"]
    raw_data_path = config["load_dataset"]["raw_dataset_path"]

    shutil.copy(initial_data_path, raw_data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    load_dataset(config_path=parsed_args.config)
