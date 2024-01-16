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

def char_gram(train_df, test_df, config):
    char_range = tuple(config['variables']['charRange'])
    n_best_factor = config['variables']['nBestFactorChar']
    lower = bool(config['variables']['lower'])
    use_LSA = bool(config['variables']['useLSA'])
    masking = bool(config['variables']['masking'])
    if masking:
        n_best_factor = 1
        vocab_word = extend_vocabulary([1,1], train_df['text'], model='word')
        print(vocab_word[:50])
        print(vocab_word[-50:])
    vocab_char = extend_vocabulary(char_range, train_df['text'], model='char-std')
    print(len(vocab_char))
    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    vectorizer_char = TfidfVectorizer(analyzer='char', ngram_range=char_range, use_idf=True,
                                      norm='l2', lowercase=lower, vocabulary=vocab_char,
                                      smooth_idf=True, sublinear_tf=True)
    print(vectorizer_char.vocabulary.index('ja'))
    n = vectorizer_char.vocabulary.index('ja')
    train_data_word = vectorizer_char.fit_transform(train_df['text']).toarray()

    print(train_data_word[:, n])
    n_best = int(len(vectorizer_char.idf_) * n_best_factor)

    idx_w = np.argsort(vectorizer_char.idf_)[:n_best]
    m = list(idx_w).index(n)
    train_data_word = train_data_word[:, idx_w]
    test_data_word = vectorizer_char.transform(test_df['text']).toarray()
    test_data_word = test_data_word[:, idx_w]

    max_abs_scaler = preprocessing.MaxAbsScaler()

    ## scale text data for word n-gram model ##
    scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
    scaled_test_data_word = max_abs_scaler.transform(test_data_word)
    print(scaled_train_data_word[:, m])
    if use_LSA:
        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components=63, algorithm='randomized', random_state=43)

        # Word
        scaled_train_data_word = svd.fit_transform(scaled_train_data_word)
        scaled_test_data_word = svd.transform(scaled_test_data_word)

    word = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    word.fit(scaled_train_data_word, train_df['author'])
    preds_char = word.predict(scaled_test_data_word)
    probas_char = word.predict_proba(scaled_test_data_word)

    return preds_char,probas_char