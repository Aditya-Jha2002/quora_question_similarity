# -*- coding: utf-8 -*-
from src import utils
from nltk.tokenize import word_tokenize
import argparse
import nltk
nltk.download("punkt")


def build_features(config_path):
    """ Script to turn the cleaned data from (../interm) into
        features ready to be trained by a model (saved in ../interim).
    """
    config = utils.read_params(config_path)
    clean_data_path = config["clean_dataset"]["clean_dataset_path"]
    feature_data_path = config["build_features"]["feature_dataset_path"]
    df = utils.get_data(clean_data_path)

    df.fillna(" ", inplace=True)

    # Get the text length feature
    df.loc[:, "question1_len"] = df.loc[:,
                                        "question1"].apply(lambda x: len(str(x)))
    df.loc[:, "question2_len"] = df.loc[:,
                                        "question2"].apply(lambda x: len(str(x)))

    # Get token out of the text
    df.loc[:, "question1_tokens"] = df.loc[:, "question1"].apply(
        lambda x: word_tokenize(str(x)))
    df.loc[:, "question2_tokens"] = df.loc[:, "question2"].apply(
        lambda x: word_tokenize(str(x)))

    # Get the number of words in the text
    df.loc[:, "question1_word_len"] = df.loc[:,
                                             "question1_tokens"].apply(lambda x: len(x))
    df.loc[:, "question2_word_len"] = df.loc[:,
                                             "question2_tokens"].apply(lambda x: len(x))

    df["diff_len"] = df["question1_len"] - df["question2_len"]

    df.dropna(inplace=True)

    len_common_words = []
    for _, row in df.iterrows():
        print(row.id)
        len_common_words.append(
            len(set(row.question1.split()).intersection(set(row.question2.split()))))
    df["len_common_words"] = len_common_words

    df.to_csv(feature_data_path, sep=",", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    build_features(config_path=parsed_args.config)
