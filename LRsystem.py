import lir
from helper_functions.split import split
from helper_functions.data_scaler import data_scaler
from helper_functions.classifiers import binary_classifier
import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import argparse
import pandas as pd
from helper_functions.df_loader import load_df
import os
import csv
from helper_functions import plotting
import torch
from transformers import BertTokenizer, RobertaTokenizer
from helper_functions.text_transformer import transform_list_of_texts
from LR_BERT import bert_model_scores
from helper_functions.classifiers import SVM_model_scores

def LR(args, config):
    # Random Seed at file level
    random_seed = 33
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Load dataframe
    full_df, config = load_df(args, config)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])
    pd.set_option('display.max_columns', None)

    model = config['variables']['model']
    n_authors = config['variables']['nAuthors']

    # Limit the authors to nAuthors
    authors = list(set(full_df.author_id))
    reduced_df = full_df.loc[full_df['author_id'].isin(authors[:n_authors])]
    add = bool(config['addAuthors'])

    if add:
        additional_df = full_df.loc[full_df['author_id'].isin(authors[n_authors:2 * n_authors])]
    df = reduced_df.copy()

    # Make a list of possible combinations of conversations when leaving one out
    convs = df['conversation'].unique()
    n_conv = len(convs)
    if bool(config['crossVal']):
        combinations = []
        for comb in itertools.combinations(convs, n_conv - 1):
            rest = list(set(convs) - set(comb))
            combinations.append([list(comb), list(rest)])
    else:
        convs_list = list(convs)
        combinations = [(convs_list[:-1],[convs_list[-1]])]

    # Initialize arrays for collecting resulting LRs
    validation_lr = np.zeros(len(combinations) * n_authors ** 2)
    additional_lr = np.zeros(len(combinations) * n_authors ** 2)
    validation_truth = np.zeros(len(combinations) * n_authors ** 2)

    for i, comb in enumerate(combinations):
        df = reduced_df.copy()

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.125, stratify=df[['author']])
        else:
            # For now only works without confusion
            train_df, test_df = split(df, 1 / n_conv, comb, confusion=False)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        calibration_df = train_df.copy()

        pd.set_option('display.max_columns', None)
        if add:
            test_df = pd.concat([test_df, additional_df[additional_df['conversation'] == comb[1][0]]])

        if config['modelType'] == 'SVM' or config['modelType'] == 'baseline':
            scaled_train_data_word, scaled_test_data_word = data_scaler(train_df, test_df, config, model='word')
            scaled_train_data_char, scaled_test_data_char = data_scaler(train_df, test_df, config, model='char-std')

        elif config['modelType'] == 'BERT':
            # Assign device and check if it is GPU or not
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Set tokenizer and tokenize training and test texts
            if config['BERT']['model'] == 'RobBERT':
                tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
            elif config['BERT']['model'] == 'BERTje':
                tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
            train_encodings, train_encodings_simple = np.array(
                transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256,
                                        device=device))
            val_encodings, val_encodings_simple = np.array(
                transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256,
                                        device=device))
            if config['BERT']['type'] == 'truncated':
                train_encodings, val_encodings = train_encodings_simple, val_encodings_simple

        conversations = list(set(train_df['conversation']))
        print(f"Test conversation: {comb[1]}")
        for suspect in range(0, len(set(train_df['author_id']))):
            train_df['h1'] = [1 if author == suspect else 0 for author in train_df['author_id']]
            test_df['h1'] = [1 if author == suspect else 0 for author in test_df['author_id']]
            calibration_df['h1'] = [1 if author == suspect else 0 for author in calibration_df['author_id']]
            calibration_scores = np.zeros(len(conversations) * n_authors)
            calibration_truth = np.zeros(len(conversations) * n_authors)

            for j, c in enumerate(conversations):
                # c is conversation in calibration set, all others go in training set
                train = train_df.index[train_df['conversation'] != c].tolist()
                calibrate = train_df.index[train_df['conversation'] == c].tolist()

                if config['modelType'] == 'SVM' or config['modelType'] == 'baseline':
                    scores = SVM_model_scores(scaled_train_data_word[train], train_df['h1'][train],
                                          scaled_train_data_word[calibrate], scaled_train_data_char[train],
                                          train_df['h1'][train], scaled_train_data_char[calibrate], model, config)

                elif config['modelType'] == 'BERT':
                    calibration_encodings = train_encodings[calibrate]
                    scores = bert_model_scores(train_encodings[train], calibration_encodings, train_df['h1'][train],
                                               train_df['h1'][calibrate], config)

                calibration_scores[j * n_authors:(j + 1) * n_authors] = scores
                calibration_truth[j * n_authors:(j + 1) * n_authors] = np.array(calibration_df['h1'][calibrate])

            # Fit bounded calibrator
            calibrator = lir.KDECalibrator(bandwidth='silverman')
            bounded_calibrator = lir.ELUBbounder(calibrator)
            bounded_calibrator.fit(calibration_scores, calibration_truth == 1)

            # Calculate scores on validation set
            if config['modelType'] == 'SVM' or config['modelType'] == 'baseline':
                validation_scores = SVM_model_scores(scaled_train_data_word, train_df['h1'],
                                                 scaled_test_data_word, scaled_train_data_char,
                                                 train_df['h1'], scaled_test_data_char, model, config)

            elif config['modelType'] == 'BERT':
                validation_scores = bert_model_scores(train_encodings, val_encodings, train_df['h1'], test_df['h1'],
                                                      config)

            # Calculate the corresponding validation LRs
            lrs_validation = bounded_calibrator.transform(validation_scores)

            # Dump all LRs and truth values into arrays
            k = i * n_authors ** 2 + suspect * n_authors
            validation_lr[k:k + n_authors] = lrs_validation[:n_authors]
            if add:
                additional_lr[k:k + n_authors] = lrs_validation[n_authors:2 * n_authors]
            validation_truth[k:k + n_authors] = np.array(test_df['h1'])[:n_authors]

            # necessary wait due to limits of LIR library
            time.sleep(0.01)
    # KDE plot
    with plotting.show() as ax:
        ones_list = np.ones(len(calibration_truth))
        d = (np.max(calibration_scores) - np.min(calibration_scores)) / 5
        ax.calibrator_fit(calibrator, score_range=[np.min(calibration_scores) - d, np.max(calibration_scores) + d],
                          resolution=1000)

        ax.score_distribution(scores=calibration_scores[calibration_truth == 1],
                              y=ones_list[calibration_truth == 1],
                              bins=np.linspace(np.min(calibration_scores) - d, np.max(calibration_scores) + d, 11),
                              weighted=True)
        ax.score_distribution(scores=calibration_scores[calibration_truth == 0],
                              y=ones_list[calibration_truth == 0] * 0,
                              bins=np.linspace(np.min(calibration_scores) - d, np.max(calibration_scores) + d, 21),
                              weighted=True)
        ax.xlabel('SVM score')
        ax.legend()
        plt.show()

    print(f"Number of authors: {n_authors}")

    cllrs = np.zeros(n_authors)
    cllrs_min = np.zeros(n_authors)
    cllrs_cal = np.zeros(n_authors)
    for j in range(n_authors):
        lr_a = np.array(
            [lr for i, lr in enumerate(validation_lr) if ((j + 1) * n_authors > i % n_authors ** 2 >= j * n_authors)])
        truth_a = np.array(
            [a for i, a in enumerate(validation_truth) if ((j + 1) * n_authors > i % n_authors ** 2 >= j * n_authors)])
        cllrs[j] = lir.metrics.cllr(lr_a, truth_a)
        cllrs_min[j] = lir.metrics.cllr_min(lr_a, truth_a)
        cllrs_cal[j] = cllrs[j] - cllrs_min[j]
        if j % 60 == 0:
            with plotting.show() as ax:
                ax.pav(lr_a, truth_a)
            plt.show()

    # Multiple box plots on one Axes
    fig, ax = plt.subplots()
    ax.boxplot([cllrs, cllrs_min, cllrs_cal], labels=['$C_{llr}$', '$C_{llr}^{min}$', '$C_{llr}^{cal}$'])
    plt.show()

    print(f"Average Cllr: {np.mean(cllrs):.3f}, Cllr_min: {np.mean(cllrs_min):.3f}, Cllr_cal: {np.mean(cllrs_cal):.3f}")
    output_file = args.output_path + os.sep + 'LR_' + args.corpus_name + ".csv"
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if config['modelType'] == 'SVM' or config['modelType'] == 'baseline':
            writer.writerow([round(np.mean(cllrs), 3), round(np.mean(cllrs_min), 3), round(np.mean(cllrs_cal), 3),
                             config['modelType'], config['variables']['nAuthors'], config['masking']['masking'],
                             config['masking']['nMasking'], config['variables']['model']])
        elif config['modelType'] == 'BERT':
            writer.writerow([round(np.mean(cllrs), 3), round(np.mean(cllrs_min), 3), round(np.mean(cllrs_cal), 3),
                             config['modelType'], config['variables']['nAuthors'], config['BERT']['model'],
                             config['BERT']['type'], config['BERT']['epochs']])

    '''
    output_file = args.output_path + os.sep + 'LRS_' + config['modelType'] + args.corpus_name + ".csv"
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(validation_lr)
        writer.writerow(validation_truth)
    '''

    h1_lrs = validation_lr[validation_truth == 1]
    h2_lrs = validation_lr[validation_truth == 0]
    freq1 = np.histogram(h1_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h1_lrs)
    freq2 = np.histogram(h2_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h2_lrs)

    print(f"H1 samples with LR < 1: {freq1[0] * 100:.3f}%, H2 samples with LR > 1: {(freq2[1] + freq2[2]) * 100:.3f}%")
    print(
        f"H1 samples with LR < 100: {(freq1[0] + freq1[1]) * 100:.3f}%, H2 samples with LR > 100: {freq2[2] * 100:.3f}%")
    if add:
        freq3 = np.histogram(additional_lr, bins=[-np.inf] + [1] + [np.inf])[0] / len(additional_lr)
        print(f"Additional samples with LR > 1: {(freq3[1]) * 100:.3f}%")
    print(f"H1 sample with lowest LR: {np.min(h1_lrs):.3f}, H2 sample with highest LR: {np.max(h2_lrs):.3f}")
    print(f"H1 sample with highest LR: {np.max(h1_lrs):.3f}, H2 sample with lowest LR: {np.min(h2_lrs):.3f}")

    # Tippet plot
    with plotting.show() as ax:
        ax.tippett(validation_lr, validation_truth)
        if add:
            lr_1 = np.log10(additional_lr)
            xplot1 = np.linspace(np.min(lr_1), np.max(lr_1), 100)
            perc1 = (sum(i >= xplot1 for i in lr_1) / len(lr_1)) * 100
            ax.plot(xplot1, perc1, color='g', label='LRs given $\mathregular{H_d}$ outside training set')
        ax.legend()
    plt.show()

    # ECE plot
    with plotting.show() as ax:
        ax.ece(validation_lr, validation_truth)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to the output")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    LR(args, config)


if __name__ == '__main__':
    main()
