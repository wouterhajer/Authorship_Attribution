import lir
from lir import *
from df_creator import *
from masking import *
from char import *
from word import *
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
    config['masking']['nBestFactorWord'] = 1

    if vocab_word == 0:
        print('hello')
        vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
    vocab_word = vocab_word[:n_masking]
    print(vocab_word)
    vocab_word = [x.lower() for x in vocab_word]
    train_df = mask(train_df, vocab_word, config)
    test_df = mask(test_df, vocab_word, config)
    full_df = mask(full_df,vocab_word,config)

# Encode author labels
label_encoder = LabelEncoder()
train_df['author'] = label_encoder.fit_transform(train_df['author'])
test_df['author'] = label_encoder.transform(test_df['author'])
full_df['author'] = label_encoder.transform(full_df['author'])
#full_df = pd.concat([train_df,test_df])

df = full_df.copy()
a = df['conversation'].unique()
combinations = []
for comb in itertools.combinations(a, 7):
    rest = list(set(a) - set(comb))
    combinations.append([list(comb), list(rest)])

# Shuffle the training data
#train_df = train_df.sample(frac=1)

# If baseline is true a top 100 word-1-gram model is used
if bool(config['baseline']):
    config['variables']['wordRange'] = [1, 1]
    vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
    config['variables']['nBestFactorWord'] = 100 / len(vocab_word)

char_range = tuple(config['variables']['charRange'])
n_best_factor = config['variables']['nBestFactorChar']
lower = bool(config['variables']['lower'])
use_LSA = bool(config['variables']['useLSA'])

lr = []
true = []
#combinations = [([1,2,4,5,6,7,8],[3])]
for i, comb in enumerate(combinations):
    print(i)
    del train_df
    del test_df
    df = full_df.copy()

    # Use random or deterministic split
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=0.125, stratify=df[['author']])
    else:
        train_df, test_df = split(df, 0.125, comb, confusion=bool(config['confusion']))

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #print(train_df,test_df)

    vocab_char = extend_vocabulary(char_range, train_df['text'], model='char-std')

    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    vectorizer_char = TfidfVectorizer(analyzer='char', ngram_range=char_range, use_idf=True,
                                      norm='l2', lowercase=lower, vocabulary=vocab_char,
                                      smooth_idf=True, sublinear_tf=True)

    train_data_word = vectorizer_char.fit_transform(train_df['text']).toarray()

    n_best = int(len(vectorizer_char.idf_) * n_best_factor)

    idx_w = np.argsort(vectorizer_char.idf_)[:n_best]

    train_data_word = train_data_word[:, idx_w]
    test_data_word = vectorizer_char.transform(test_df['text']).toarray()
    test_data_word = test_data_word[:, idx_w]

    # Choose scaler
    max_abs_scaler = preprocessing.MaxAbsScaler()
    # max_abs_scaler = preprocessing.MinMaxScaler()

    ## scale text data for char n-gram model ##
    scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
    scaled_test_data_word = max_abs_scaler.transform(test_data_word)

    num_texts = np.zeros(len(train_data_word[0]))
    for i in range(len(train_data_word)):
        num_texts += np.array([1 if element > 0 else 0 for element in scaled_train_data_word[i]])

    if use_LSA:
        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components=63, algorithm='randomized', random_state=43)

        # Word
        scaled_train_data_word = svd.fit_transform(scaled_train_data_word)
        scaled_test_data_word = svd.transform(scaled_test_data_word)

    conversations = list(set(train_df['conversation']))
    lr_split = []
    true_split = []
    for suspect in range(0, len(set(train_df['author']))):
        train_df['h1'] = [1 if author == suspect else 0 for author in train_df['author']]

        h1_cali = []
        h2_cali = []

        for c in conversations:
            char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                                  gamma='auto')))
            train = train_df.index[train_df['conversation'] != c].tolist()

            calibrate = train_df.index[train_df['conversation'] == c].tolist()

            char.fit(scaled_train_data_word[train], train_df['h1'][train])

            probas_char = char.predict_proba(scaled_train_data_word[calibrate])

            h1_cali.extend([probas_char[i, 0] for i in range(len(probas_char)) if list(train_df['h1'][calibrate])[i] == 1])
            h2_cali.extend([probas_char[i, 0] for i in range(len(probas_char)) if list(train_df['h1'][calibrate])[i] == 0])

        char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                              gamma='auto')))

        char.fit(scaled_train_data_word, train_df['h1'])
        probas_val = char.predict_proba(scaled_test_data_word)

        scorer = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                                gamma='auto')))
        calibrator = lir.KDECalibrator(bandwidth='silverman')
        scorer.fit(scaled_train_data_word, train_df['h1'])
        dissimilarity_scores_test = scorer.predict_proba(scaled_test_data_word)

        dissimilarity_scores_train = np.array(h1_cali + h2_cali)

        hypothesis_train = np.array(['H1'] * len(h1_cali) + ['H2'] * len(h2_cali))
        calibrator.fit(dissimilarity_scores_train, hypothesis_train == 'H1')


        bounded_calibrator = lir.ELUBbounder(calibrator)

        lrs_train = bounded_calibrator.fit_transform(dissimilarity_scores_train, hypothesis_train == 'H1')
        lrs_test = bounded_calibrator.transform(dissimilarity_scores_test[:,0])

        lr.extend(lrs_test)
        true.extend([1 if author == suspect else 0 for author in test_df['author']])
        lr_split.extend(lrs_test)
        true_split.extend([1 if author == suspect else 0 for author in test_df['author']])
        time.sleep(0.1)
    index = [i for i in range(len(true_split)) if true_split[i] == 1]
    print(np.array(lr_split)[index])
    cllr = lir.metrics.cllr(np.array(lr_split), np.array(true_split))
    cllr_min = lir.metrics.cllr_min(np.array(lr_split), np.array(true_split))
    cllr_cal = cllr - cllr_min
    print(f"Test conversation: {comb[1]}")
    print(f"Cllr: {cllr}, Cllr_min: {cllr_min}, Cllr_cal: {cllr_cal}")
    with lir.plotting.show() as ax:
        ax.calibrator_fit(calibrator, score_range=[0, 1])
        ax.score_distribution(scores=dissimilarity_scores_train, y=(hypothesis_train == 'H1') * 1,
                              bins=np.linspace(0, 1, 20), weighted=True)
        ax.xlabel('SVM score')
        H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='$H_1$-true')
        H1_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='$H_1$-true')

    plt.show()

with lir.plotting.show() as ax:
    ax.calibrator_fit(calibrator, score_range=[0, 1])
    ax.score_distribution(scores=dissimilarity_scores_train, y=(hypothesis_train == 'H1') * 1,
                          bins=np.linspace(0, 1, 20), weighted=True)
    ax.xlabel('SVM score')
    H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='$H_1$-true')
    H1_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='$H_1$-true')

plt.show()
plt.scatter(dissimilarity_scores_train, np.log10(lrs_train))
plt.show()

plt.scatter(true, np.log10(lr))
plt.show()
df = df.sort_index(axis=0)
print(train_df)
index = [i for i in range(len(true)) if true[i] == 1]
print(np.array(lr)[index])
cllr = lir.metrics.cllr(np.array(lr), np.array(true))
cllr_min = lir.metrics.cllr_min(np.array(lr), np.array(true))
cllr_cal = cllr-cllr_min
print(f"Cllr: {cllr}, Cllr_min: {cllr_min}, Cllr_cal: {cllr_cal}")
