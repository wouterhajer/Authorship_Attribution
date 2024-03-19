import lir
from lir import *
from df_creator import *
from masking import *
from data_scaler import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
import numpy as np
import matplotlib.patches as mpatches
import time
import itertools
from multiclass_classifier import binary_classifier


def model_scores(train_word, truth_word, test_word, train_char, truth_char, test_char, model):
    if model == 'both':
        word = binary_classifier(train_word, truth_word, test_word)
        char = binary_classifier(train_char, truth_char, test_char)
        return (word + char) / 2
    elif model == 'word':
        return binary_classifier(train_word, truth_word, test_word)
    elif model == 'char':
        return binary_classifier(train_char, truth_char, test_char)


with open('config.json') as f:
    config = json.load(f)

# Random Seed at file level
random_seed = 33
np.random.seed(random_seed)
random.seed(random_seed)

full_df, train_df, test_df, vocab_word = create_df('txt', config, p_test=0.125)

# If masking is turned on replace all words outside top n_masking with asterisks
n_masking = config['masking']['nMasking']
if bool(config['masking']['masking']):
    config['variables']['nBestFactorWord'] = 1
    config['variables']['nBestFactorChar'] = 1
    if vocab_word == 0:
        vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
    vocab_masking = vocab_word[:n_masking]
    print(vocab_masking)
    vocab_masking = [x.lower() for x in vocab_masking]
    full_df = mask(full_df, vocab_masking, config)

# Encode author labels
label_encoder = LabelEncoder()
full_df['author'] = label_encoder.fit_transform(full_df['author'])
# full_df = pd.concat([train_df,test_df])

# Shuffle the training data
# train_df = train_df.sample(frac=1)

# If baseline is true a top 100 word-1-gram model is used
if bool(config['baseline']):
    config['variables']['wordRange'] = [1, 1]
    config['variables']['model'] = "word"
    config['variables']['useLSA'] = 0

char_range = tuple(config['variables']['charRange'])
word_range = tuple(config['variables']['wordRange'])
n_best_factor = config['variables']['nBestFactorChar']
lower = bool(config['variables']['lower'])
use_LSA = bool(config['variables']['useLSA'])
model = config['variables']['model']
n_authors = config['variables']['nAuthors']
cllr_avg = np.zeros(4)

# Limit the authors to nAuthors
authors = list(set(full_df.author))
reduced_df = full_df.loc[full_df['author'].isin(authors[:n_authors])]
additional_df = full_df.loc[full_df['author'].isin(authors[n_authors:2 * n_authors])]

df = reduced_df.copy()
a = df['conversation'].unique()
combinations = []
for comb in itertools.combinations(a, 7):
    rest = list(set(a) - set(comb))
    combinations.append([list(comb), list(rest)])

#combinations = [([1, 2, 4, 5, 6, 7, 8], [3])]
validation_lr = np.zeros(len(combinations) * n_authors ** 2)
additional_lr = np.zeros(len(combinations) * n_authors ** 2)
validation_truth = np.zeros(len(combinations) * n_authors ** 2)
for i, comb in enumerate(combinations):
    print(i)
    del train_df
    del test_df
    df = reduced_df.copy()

    # Use random or deterministic split
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=0.125, stratify=df[['author']])
    else:
        # For now only works without confusion
        train_df, test_df = split(df, 0.125, comb, confusion=False)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    test_df = pd.concat([test_df, additional_df[additional_df['conversation'] == comb[1][0]]])

    scaled_train_data_word, scaled_test_data_word = data_scaler(train_df, test_df, config, model='word')
    scaled_train_data_char, scaled_test_data_char = data_scaler(train_df, test_df, config, model='char-std')

    conversations = list(set(train_df['conversation']))
    lr_split = []
    true_split = []
    print(f"Test conversation: {comb[1]}")
    for suspect in range(0, len(set(train_df['author']))):
        train_df['h1'] = [1 if author == suspect else 0 for author in train_df['author']]
        test_df['h1'] = [1 if author == suspect else 0 for author in test_df['author']]

        calibration_scores = np.zeros(len(conversations) * n_authors)
        calibration_truth = np.zeros(len(conversations) * n_authors)
        for j, c in enumerate(conversations):
            # c is conversation in calibration set, all others go in training set
            train = train_df.index[train_df['conversation'] != c].tolist()
            calibrate = train_df.index[train_df['conversation'] == c].tolist()
            """
            scores_word = binary_classifier(scaled_train_data_word[train], train_df['h1'][train],
                                            scaled_train_data_word[calibrate])
            scores_char = binary_classifier(scaled_train_data_char[train], train_df['h1'][train],
                                            scaled_train_data_char[calibrate])
            """
            scores = model_scores(scaled_train_data_word[train], train_df['h1'][train],
                                  scaled_train_data_word[calibrate], scaled_train_data_char[train],
                                  train_df['h1'][train], scaled_train_data_char[calibrate], model)

            calibration_scores[j * n_authors:(j + 1) * n_authors] = scores
            calibration_truth[j * n_authors:(j + 1) * n_authors] = np.array(train_df['h1'][calibrate])
        """
        scores_word = binary_classifier(scaled_train_data_word, train_df['h1'],
                                        scaled_test_data_word)
        scores_char = binary_classifier(scaled_train_data_char, train_df['h1'],
                                        scaled_test_data_char)
        """

        validation_scores = model_scores(scaled_train_data_word, train_df['h1'],
                                         scaled_test_data_word, scaled_train_data_char,
                                         train_df['h1'], scaled_test_data_char, model)

        calibrator = lir.KDECalibrator(bandwidth='silverman')  # [0.01,0.1]
        calibrator.fit(calibration_scores, calibration_truth == 1)
        bounded_calibrator = lir.ELUBbounder(calibrator)
        bounded_calibrator.fit(calibration_scores, calibration_truth == 1)
        lrs_validation = bounded_calibrator.transform(validation_scores)

        k = i * n_authors ** 2 + suspect * n_authors
        validation_lr[k:k + n_authors] = lrs_validation[:n_authors]
        additional_lr[k:k + n_authors] = lrs_validation[n_authors:2*n_authors]
        validation_truth[k:k + n_authors] = np.array(test_df['h1'])[:n_authors]

        time.sleep(0.01)
        cllr = lir.metrics.cllr(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
        cllr_min = lir.metrics.cllr_min(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
        cllr_cal = cllr - cllr_min
        cllr_avg = cllr_avg + np.array([cllr, cllr_min, cllr_cal, 1])
        print(f"Average Cllr: {cllr_avg[0] / cllr_avg[3]:.3f}, Cllr_min: {cllr_avg[1] / cllr_avg[3]:.3f}\
                , Cllr_cal: {cllr_avg[2] / cllr_avg[3]:.3f}")
        ones_list = np.ones(len(calibration_truth))
        """
        with lir.plotting.show() as ax:
            ax.calibrator_fit(calibrator, score_range=[0, 1], resolution = 1000)
            ax.score_distribution(scores=calibration_scores[calibration_truth == 1],
                                  y=ones_list[calibration_truth == 1],
                                  bins=np.linspace(0, 1, 9), weighted=True)
            ax.score_distribution(scores=calibration_scores[calibration_truth == 0],
                                  y=ones_list[calibration_truth == 0]*0,
                                  bins=np.linspace(0, 1, 41), weighted=True)
            ax.xlabel('SVM score')
            H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='$H_1$-true')
            H2_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='$H_2$-true')
            ax.legend()
            plt.show()

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

freq1 = np.histogram(h1_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h1_lrs)
freq2 = np.histogram(h2_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h2_lrs)
print(f"H1 samples with LR < 1: {freq1[0] * 100:.3f}%, H2 samples with LR > 1: {(freq2[1] + freq2[2]) * 100:.3f}%")
print(f"H1 samples with LR < 100: {(freq1[0] + freq1[1]) * 100:.3f}%, H2 samples with LR > 100: {freq2[2] * 100:.3f}%")
print(f"H1 sample with lowest LR: {np.min(h1_lrs):.3f}, H2 sample with highest LR: {np.max(h2_lrs):.3f}")
print(f"H1 sample with highest LR: {np.max(h1_lrs):.3f}, H2 sample with lowest LR: {np.min(h2_lrs):.3f}")
"""
for j in range(n_authors):
    avg = np.zeros(n_authors)
    for i in range(len(combinations)):
        avg += np.log10(additional_lr[i*n_authors**2+j*n_authors:i*n_authors**2+(j+1)*n_authors])
    avg /= len(combinations)
    x = np.linspace(51,100,50)
    plt.scatter(x,avg)
    plt.show()
"""
with lir.plotting.show() as ax:
    ax.tippett(validation_lr, validation_truth)
    lr_1 = np.log10(additional_lr)
    xplot1 = np.linspace(np.min(lr_1), np.max(lr_1), 100)
    perc1 = (sum(i >= xplot1 for i in lr_1) / len(lr_1)) * 100
    ax.plot(xplot1, perc1, color='g', label='LRs given $\mathregular{H_2}$ outside training set')
    ax.legend()
plt.show()

plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 1],
            np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 1])
plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 0],
            np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 0])
plt.show()
