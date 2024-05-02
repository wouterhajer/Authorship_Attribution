import time
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from split import split
import itertools
from ngram import ngram
import argparse
from df_loader import load_df
import pandas as pd

def average_f1(args, config):
    start = time.time()

    # Random Seed at file level
    random_seed = 37
    np.random.seed(random_seed)
    random.seed(random_seed)

    full_df,config = load_df(args,config)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    author_ids = list(set(full_df['author_id']))
    df = full_df.loc[full_df['author_id'].isin(author_ids[:config['variables']['nAuthors']])]

    # Find all conversation numbers and make all combinations of 6 in train set, 2 in test set
    a = df['conversation'].unique()
    combinations = []
    for comb in itertools.combinations(a, len(a)-2):
        rest = list(set(a) - set(comb))
        combinations.append([list(comb), list(rest)])

    score = 0
    score_partner = 0
    score_rest = 0
    f1_total = 0

    # For every combination of conversations in train/test set, calculate the scores for true author,
    # conversation partner and other speakers
    for i, comb in enumerate(combinations):
        print(i)

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
        else:
            train_df, test_df = split(df, 0.25, comb, confusion=bool(config['confusion']))
        pd.set_option('display.max_columns', None)
        #print(train_df)
        #print(test_df)
        # Train SVMs, calculate predictions
        avg_preds, preds_char, preds_word, test_authors = ngram(train_df, test_df, config)

        # Inverse label encodings
        avg_preds = label_encoder.inverse_transform(avg_preds)
        preds_char = label_encoder.inverse_transform(preds_char)
        preds_word = label_encoder.inverse_transform(preds_word)
        test_authors = label_encoder.inverse_transform(test_authors)
        print(test_authors)
        print(avg_preds)
        print(list(test_df['author']))
        # When using baseline, only word prediction counts
        if bool(config['baseline']):
            avg_preds = preds_word

        # Calculate the scores
        if args.corpus_name == 'Frida':
            for j in range(len(test_df['author'])):
                if test_authors[j] == avg_preds[j]:
                    score += 1
                elif test_authors[j] == avg_preds[j] - 1 and test_authors[j] % 2 == 1:
                    score_partner += 1
                elif test_authors[j] == avg_preds[j] + 1 and test_authors[j] % 2 == 0:
                    score_partner += 1
                else:
                    score_rest += 1
        # Calculate the scores
        elif args.corpus_name == 'abc_nl1':
            print('hello')
            for j in range(len(test_df['author'])):
                if test_authors[j] == avg_preds[j]:
                    score += 1
                elif avg_preds[j] % 2 == 0 and test_authors[j] % 2 == 1:
                    score_partner += 1
                elif avg_preds[j] % 2 == 1 and test_authors[j] % 2 == 0:
                    score_partner += 1
                else:
                    score_rest += 1

        n_prob = len(test_authors)
        n_auth = len(set(test_authors))
        f1 = f1_score(test_authors, avg_preds, average='macro')

        f1_total += f1
        # Print the scores at this iteration
        print("Score = {:.4f}, random chance = {:.4f} ".format(score / n_prob / (i + 1), 1 / n_auth))
        print("Score partner = {:.4f}, random chance = {:.4f} ".format(score_partner / n_prob / (i + 1), 1 / n_auth))
        print("Score rest = {:.4f}, random chance = {:.4f} ".format(score_rest / n_prob / (i + 1), 1 - 2 / n_auth))
        print("F1-score = {:.4f}".format(f1))
        print("Average F1-score = {:.4f}".format(f1_total / (i + 1)))

    print('Included authors: ' + str(len(set(test_authors))))
    # Print duration
    print('Total time: ' + str(time.time() - start) + ' seconds')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to output folder")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)

    average_f1(args, config)


if __name__ == '__main__':
    main()
