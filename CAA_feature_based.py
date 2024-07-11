import time
import json
import random
import argparse
import os
import csv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from helper_functions.split import split, combinations
from helper_functions.classifiers import feature_based_classification
from helper_functions.df_loader import load_df
from helper_functions.group_scores import group_scores


def CAA_feature_based(args, config):
    start = time.time()

    # Random Seed at file level
    random_seed = 37
    np.random.seed(random_seed)
    random.seed(random_seed)

    full_df, config = load_df(args, config)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    author_ids = list(set(full_df['author_id']))
    df = full_df.loc[full_df['author_id'].isin(author_ids[:config['variables']['nAuthors']])]

    # Make a list of possible combinations of conversations when leaving one out
    convs = df['conversation'].unique()
    combs = combinations(convs, bool(config['crossVal']))

    score = 0
    score_partner = 0
    score_rest = 0
    f1_total = 0

    # For every combination of conversations in train/test set, calculate the scores for true author,
    # conversation partner and other speakers
    for i, comb in enumerate(combs):
        print(i)

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
        else:
            train_df, test_df = split(df, 0.25, comb, confusion=bool(config['confusion']))
        pd.set_option('display.max_columns', None)

        # Train models, calculate predictions
        avg_preds, test_authors = feature_based_classification(train_df, test_df, config)

        # Inverse label encodings
        avg_preds = label_encoder.inverse_transform(avg_preds)
        test_authors = label_encoder.inverse_transform(test_authors)

        # Calculate the scores
        score_i, score_partner_i, score_rest_i = group_scores(test_authors, avg_preds, args)
        score += score_i
        score_partner += score_partner_i
        score_rest += score_rest_i

        n_auth = len(set(test_authors))
        f1 = f1_score(test_authors, avg_preds, average='macro')
        f1_total += f1

        # Print the scores at this iteration
        print("Score = {:.4f}, random chance = {:.4f} ".format(score / (i + 1), 1 / n_auth))
        print("Score partner = {:.4f}, random chance = {:.4f} ".format(score_partner / (i + 1), 1 / n_auth))
        print("Score rest = {:.4f}, random chance = {:.4f} ".format(score_rest  / (i + 1), 1 - 2 / n_auth))
        print("F1-score = {:.4f}".format(f1))
        print("Average F1-score = {:.4f}".format(f1_total / (i + 1)))

    print('Included authors: ' + str(len(set(test_authors))))
    # Print duration
    print('Total time: ' + str(time.time() - start) + ' seconds')
    output_file = args.output_path + os.sep + 'main_' + args.corpus_name + ".csv"
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(f1_total/(i +1),3), round(score / (i + 1),3),
                         round(score_partner / (i + 1),3), round(score_rest / (i + 1),3),
                         config['confusion'],config['variables']['nAuthors'], config['baseline'],
                         config['masking']['masking'], config['masking']['nMasking'], config['variables']['model']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to output folder")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)

    CAA_feature_based(args, config)


if __name__ == '__main__':
    main()
