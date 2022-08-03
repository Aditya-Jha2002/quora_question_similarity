# -*- coding: utf-8 -*-
import argparse
import os
from src import utils

from src.data.preprocessing import preprocess_text_column

def remove_space(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)


def clean_dataset(config_path):
    """ Runs data preprocessing scripts to turn raw data from (../raw) into
        cleaned and pre-processed data ready to be feature engineered on (saved in ../interim).
    """
    config = utils.read_params(config_path)
    df = utils.get_data(config_path)

    #Drop all of the null values
    df.dropna(inplace=True)
    
    #Preprocess the question1 and question2 columns
    preprocess_text_column(df, "question1")
    preprocess_text_column(df, "question2")

    #Storing the dataframe to interim folder
    df.to_csv(config["clean_dataset"]["clean_dataset_path"], sep=",", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    clean_dataset(config_path = parsed_args.config) 