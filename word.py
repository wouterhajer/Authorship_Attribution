from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
import numpy as np
from vocabulary import *


def word_gram(train_df, test_df, config):
    word_range = tuple(config['variables']['wordRange'])
    n_best_factor = config['variables']['nBestFactorWord']
    lower = bool(config['variables']['lower'])
    use_LSA = bool(config['variables']['useLSA'])

    vocab_word = extend_vocabulary(word_range, train_df['text'], model='word')

    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    vectorizer_word = TfidfVectorizer(analyzer='word', ngram_range=word_range, use_idf=True,
                                      norm='l2', lowercase=lower, vocabulary=vocab_word,
                                      smooth_idf=True, sublinear_tf=True)

    train_data_word = vectorizer_word.fit_transform(train_df['text']).toarray()

    n_best = int(len(vectorizer_word.idf_) * n_best_factor)
    idx_w = np.argsort(vectorizer_word.idf_)[:n_best]

    train_data_word = train_data_word[:, idx_w]
    test_data_word = vectorizer_word.transform(test_df['text']).toarray()
    test_data_word = test_data_word[:, idx_w]

    max_abs_scaler = preprocessing.MaxAbsScaler()

    ## scale text data for word n-gram model ##
    scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
    scaled_test_data_word = max_abs_scaler.transform(test_data_word)

    num_texts = np.zeros(len(train_data_word[0]))
    for i in range(len(train_data_word)):
        num_texts += np.array([1 if element > 0 else 0 for element in scaled_train_data_word[i]])

    perc_texts = num_texts / len(train_data_word)
    # print(np.array(scaled_train_data_word)[0,-30:])
    # print((np.array(scaled_train_data_word)*np.exp(0.05*(1-1/perc_texts)))[0,-)
    #scaled_train_data_word = list(np.array(scaled_train_data_word) * np.exp(0.1*(2-1/perc_texts)))

    if use_LSA:
        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components=500, algorithm='randomized', random_state=42)

        # Word
        scaled_train_data_word = svd.fit_transform(scaled_train_data_word)
        scaled_test_data_word = svd.transform(scaled_test_data_word)

    word = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    word.fit(scaled_train_data_word, train_df['author'])
    preds_word = word.predict(scaled_test_data_word)
    probas_word = word.predict_proba(scaled_test_data_word)

    return preds_word,probas_word