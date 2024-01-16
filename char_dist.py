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


def char_dist_gram(train_df, test_df, config):
    char_dist_range = tuple(config['variables']['charDistRange'])
    n_best_factor = config['variables']['nBestFactorChar']
    lower = bool(config['variables']['lower'])
    use_LSA = bool(config['variables']['useLSA'])
    vocab_char_dist = extend_vocabulary(char_dist_range, train_df['text'], model='char-dist')

    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    vectorizer_char_dist = TfidfVectorizer(analyzer='char', ngram_range=char_dist_range, use_idf=True,
                                           norm='l2', lowercase=lower, vocabulary=vocab_char_dist,
                                           smooth_idf=True,sublinear_tf=True)

    train_data = vectorizer_char_dist.fit_transform(train_df['text']).toarray()

    n_best = int(len(vectorizer_char_dist.idf_) * n_best_factor)
    idx_w = np.argsort(vectorizer_char_dist.idf_)[:n_best]

    train_data = train_data[:, idx_w]
    test_data = vectorizer_char_dist.transform(test_df['text']).toarray()
    test_data = test_data[:, idx_w]

    max_abs_scaler = preprocessing.MaxAbsScaler()

    ## scale text data for word n-gram model ##
    scaled_train_data = max_abs_scaler.fit_transform(train_data)
    scaled_test_data = max_abs_scaler.transform(test_data)

    if use_LSA:
        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components=20, algorithm='randomized', random_state=43)

        # Word
        scaled_train_data = svd.fit_transform(scaled_train_data)
        scaled_test_data = svd.transform(scaled_test_data)

    char_dist = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                               gamma='auto')))
    char_dist.fit(scaled_train_data, train_df['author'])
    preds_char_dist = char_dist.predict(scaled_test_data)
    probas_char_dist = char_dist.predict_proba(scaled_test_data)

    return preds_char_dist, probas_char_dist
