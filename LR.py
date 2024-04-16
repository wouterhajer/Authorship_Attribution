import lir
from data_scaler import data_scaler
from split import split
from multiclass_classifier import binary_classifier
import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import time
import itertools
import argparse
import pandas as pd
from df_loader import load_df
import os
import csv
import plotting


def model_scores(train_word, truth_word, test_word, train_char, truth_char, test_char, model):
    if model == 'both':
        word = binary_classifier(train_word, truth_word, test_word)
        char = binary_classifier(train_char, truth_char, test_char)
        return (word + char) / 2
    elif model == 'word':
        return binary_classifier(train_word, truth_word, test_word)
    elif model == 'char':
        return binary_classifier(train_char, truth_char, test_char)


def LR(args,config):
    # Random Seed at file level
    random_seed = 33
    np.random.seed(random_seed)
    random.seed(random_seed)

    full_df, config = load_df(args,config)
    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])
    pd.set_option('display.max_columns', None)

    model = config['variables']['model']
    n_authors = config['variables']['nAuthors']
    cllr_avg = np.zeros(4)

    # Limit the authors to nAuthors
    authors = list(set(full_df.author_id))
    reduced_df = full_df.loc[full_df['author_id'].isin(authors[:n_authors])]
    additional_df = full_df.loc[full_df['author_id'].isin(authors[n_authors:2 * n_authors])]

    df = reduced_df.copy()
    a = df['conversation'].unique()
    combinations = []
    for comb in itertools.combinations(a, 7):
        rest = list(set(a) - set(comb))
        combinations.append([list(comb), list(rest)])

    #combinations = [([1, 2, 4, 3, 6, 7, 8], [5])]
    validation_lr = np.zeros(len(combinations) * n_authors ** 2)
    additional_lr = np.zeros(len(combinations) * n_authors ** 2)
    validation_truth = np.zeros(len(combinations) * n_authors ** 2)
    for i, comb in enumerate(combinations):
        print(i)
        df = reduced_df.copy()

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.125, stratify=df[['author']])
        else:
            # For now only works without confusion
            train_df, test_df = split(df, 0.125, comb, confusion=False)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        n = 1
        calibration_df = train_df.copy()

        new_df = pd.DataFrame(columns=['text', 'author', 'conversation', 'author_id'])

        for index, row in calibration_df.iterrows():
            splits = row['text'].split('.')
            
            for k in range(n):
                m = len(splits) / n
                if n == 1:
                    m = 0
                m = 0
                #inds = set(random.sample(list(range(len(splits))), int(m)))
                inds = list(range(int(k*m), int((k+1)*m)))

                text = [n for i, n in enumerate(splits) if i not in inds]
                #text = splits
                new_df = pd.concat([new_df, pd.DataFrame(
                    {'text': ['.'.join(text)], 'author': [row['author']],
                     'conversation': [row['conversation']], 'author_id': [row['author_id']]})],
                                   ignore_index=True)

        new_df = new_df.reset_index(drop=True)
        calibration_df = new_df

        pd.set_option('display.max_columns', None)
        test_df = pd.concat([test_df, additional_df[additional_df['conversation'] == comb[1][0]]])
        #add_df = additional_df[additional_df['conversation'].isin(comb[0])]
        scaled_train_data_word, scaled_test_data_word = data_scaler(train_df, test_df, config, model='word')
        scaled_train_data_char, scaled_test_data_char = data_scaler(train_df, test_df, config, model='char-std')

        a, scaled_cali_data_word = data_scaler(train_df, calibration_df, config, model='word')
        a, scaled_cali_data_char = data_scaler(train_df, calibration_df, config, model='char-std')

        #a, scaled_add_data_word = data_scaler(train_df, add_df, config, model='word')
        #a, scaled_add_data_char = data_scaler(train_df, add_df, config, model='char-std')

        conversations = list(set(train_df['conversation']))
        print(f"Test conversation: {comb[1]}")
        for suspect in range(0, len(set(train_df['author_id']))):
            train_df['h1'] = [1 if author == suspect else 0 for author in train_df['author_id']]
            test_df['h1'] = [1 if author == suspect else 0 for author in test_df['author_id']]
            calibration_df['h1'] = [1 if author == suspect else 0 for author in calibration_df['author_id']]
            calibration_scores = np.zeros(n*len(conversations) * n_authors)
            calibration_truth = np.zeros(n*len(conversations) * n_authors)
            for j, c in enumerate(conversations):
                # c is conversation in calibration set, all others go in training set
                train = train_df.index[train_df['conversation'] != c].tolist()
                calibrate = train_df.index[train_df['conversation'] == c].tolist()
                calibrate2 = calibration_df.index[calibration_df['conversation'] == c].tolist()
                scores = model_scores(scaled_train_data_word[train], train_df['h1'][train],
                                      scaled_cali_data_word[calibrate2], scaled_train_data_char[train],
                                      train_df['h1'][train], scaled_cali_data_char[calibrate2], model)

                #scores = np.log(scores/(1-scores))

                calibration_scores[n*j * n_authors:n*(j + 1) * n_authors] = scores
                calibration_truth[n*j * n_authors:n*(j + 1) * n_authors] = np.array(calibration_df['h1'][calibrate2])

            validation_scores = model_scores(scaled_train_data_word, train_df['h1'],
                                             scaled_test_data_word, scaled_train_data_char,
                                             train_df['h1'], scaled_test_data_char, model)

            #validation_scores = np.log(validation_scores / (1 - validation_scores))

            calibrator = lir.KDECalibrator(bandwidth='silverman')
            calibrator.fit(calibration_scores, calibration_truth == 1)

            bounded_calibrator = lir.ELUBbounder(calibrator)
            bounded_calibrator.fit(calibration_scores, calibration_truth == 1)
            lrs_validation = bounded_calibrator.transform(validation_scores)

            k = i * n_authors ** 2 + suspect * n_authors
            validation_lr[k:k + n_authors] = lrs_validation[:n_authors]
            additional_lr[k:k + n_authors] = lrs_validation[n_authors:2*n_authors]
            validation_truth[k:k + n_authors] = np.array(test_df['h1'])[:n_authors]

            # necessary wait due to limits of LIR library
            time.sleep(0.01)

            # Calculate the current Cllrs
            cllr = lir.metrics.cllr(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
            cllr_min = lir.metrics.cllr_min(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
            cllr_cal = cllr - cllr_min
            cllr_avg = cllr_avg + np.array([cllr, cllr_min, cllr_cal, 1])
            #print(f"Average Cllr: {cllr_avg[0] / cllr_avg[3]:.3f}, Cllr_min: {cllr_avg[1] / cllr_avg[3]:.3f}\
            #        , Cllr_cal: {cllr_avg[2] / cllr_avg[3]:.3f}")
            ones_list = np.ones(len(calibration_truth))

        with plotting.show() as ax:
            d = (np.max(calibration_scores) - np.min(calibration_scores))/8
            ax.calibrator_fit(calibrator, score_range=[np.min(calibration_scores)-d, np.max(calibration_scores)+d], resolution=1000)

            ax.score_distribution(scores=calibration_scores[calibration_truth == 1],
                                  y=ones_list[calibration_truth == 1],
                                  bins=np.linspace(np.min(calibration_scores)-d, np.max(calibration_scores)+d, 11), weighted=True)
            ax.score_distribution(scores=calibration_scores[calibration_truth == 0],
                                  y=ones_list[calibration_truth == 0] * 0,
                                  bins=np.linspace(np.min(calibration_scores)-d, np.max(calibration_scores)+d, 21), weighted=True)
            ax.xlabel('SVM score')
            H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='$H_1$-true')
            H2_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='$H_2$-true')
            ax.legend()
            plt.show()

    """
    with lir.plotting.show() as ax:
        ax.tippett(validation_lr[i*n_authors**2:(i+1)*n_authors**2], validation_truth[i*n_authors**2:(i+1)*n_authors**2])
    plt.show()
    """
    print(f"Nauthors: {n_authors}")

    h1_lrs = validation_lr[validation_truth == 1]
    h2_lrs = validation_lr[validation_truth == 0]
    cllr = lir.metrics.cllr(validation_lr, validation_truth)
    cllr_min = lir.metrics.cllr_min(validation_lr, validation_truth)
    cllr_cal = cllr - cllr_min
    print(f"Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")
    print(f"Average Cllr: {cllr_avg[0] / cllr_avg[3]:.3f}, Cllr_min: {cllr_avg[1] / cllr_avg[3]:.3f}\
                , Cllr_cal: {cllr_avg[2] / cllr_avg[3]:.3f}")
    cllrs = np.zeros(n_authors)
    cllrs_min = np.zeros(n_authors)
    cllrs_cal = np.zeros(n_authors)
    for j in range(n_authors):
        lr_a = np.array([lr for i,lr in enumerate(validation_lr) if ((j + 1) * n_authors > i % n_authors**2 >= j * n_authors)])
        truth_a = np.array([a for i,a in enumerate(validation_truth) if ((j + 1) * n_authors > i % n_authors**2 >= j * n_authors)])
        cllrs[j] = lir.metrics.cllr(lr_a, truth_a)
        cllrs_min[j] = lir.metrics.cllr_min(lr_a, truth_a)
        cllrs_cal[j] = cllrs[j] - cllrs_min[j]
        if j % 60 == 0:
            with lir.plotting.show() as ax:
                ax.pav(lr_a, truth_a)
            plt.show()

    plt.plot(cllrs)
    plt.plot(cllrs_min)
    plt.plot(cllrs_cal)
    plt.show()
    print(f"Average Cllr: {np.mean(cllrs):.3f}, Cllr_min: {np.mean(cllrs_min):.3f}, Cllr_cal: {np.mean(cllrs_cal):.3f}")
    output_file = args.output_path + os.sep + 'LR_' + args.corpus_name + ".csv"
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(cllr, 3), round(cllr_min, 3),round(cllr_cal, 3), round(np.mean(cllrs_min),3),\
                         round(np.mean(cllrs_cal),3),config['variables']['nAuthors'], \
                         config['masking']['masking'], config['masking']['nMasking'], config['variables']['model']])


    freq1 = np.histogram(h1_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h1_lrs)
    freq2 = np.histogram(h2_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h2_lrs)
    print(f"H1 samples with LR < 1: {freq1[0] * 100:.3f}%, H2 samples with LR > 1: {(freq2[1] + freq2[2]) * 100:.3f}%")
    print(f"H1 samples with LR < 100: {(freq1[0] + freq1[1]) * 100:.3f}%, H2 samples with LR > 100: {freq2[2] * 100:.3f}%")
    print(f"H1 sample with lowest LR: {np.min(h1_lrs):.3f}, H2 sample with highest LR: {np.max(h2_lrs):.3f}")
    print(f"H1 sample with highest LR: {np.max(h1_lrs):.3f}, H2 sample with lowest LR: {np.min(h2_lrs):.3f}")

    with lir.plotting.show() as ax:
        ax.tippett(validation_lr, validation_truth)
        lr_1 = np.log10(additional_lr)
        xplot1 = np.linspace(np.min(lr_1), np.max(lr_1), 100)
        perc1 = (sum(i >= xplot1 for i in lr_1) / len(lr_1)) * 100
        ax.plot(xplot1, perc1, color='g', label='LRs given $\mathregular{H_2}$ outside training set')
        ax.legend()
    plt.show()

    with lir.plotting.show() as ax:
        ax.pav(validation_lr, validation_truth)
    plt.show()

    with lir.plotting.show() as ax:
        ax.ece(validation_lr, validation_truth)
    plt.show()

    plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 1],
                np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 1])
    plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 0],
                np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 0])
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
