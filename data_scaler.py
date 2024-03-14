from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from vocabulary import extend_vocabulary
import numpy as np

def data_scaler(train_df,test_df,config,model = 'char-std'):
    if model == 'char-std':
        range = tuple(config['variables']['charRange'])
        n_best_factor = config['variables']['nBestFactorChar']
    elif model == 'word':
        range = tuple(config['variables']['wordRange'])
        n_best_factor = config['variables']['nBestFactorWord']
    else:
        print('No model specified')
    lower = bool(config['variables']['lower'])
    use_LSA = bool(config['variables']['useLSA'])

    vocab = extend_vocabulary(range, train_df['text'], model=model)

    ## initialize tf-idf vectorizer for n-gram model (captures content) ##
    vectorizer = TfidfVectorizer(analyzer=model[:4], ngram_range=range, use_idf=True,
                                      norm='l2', lowercase=lower, vocabulary=vocab,
                                      smooth_idf=True, sublinear_tf=True)

    train_data = vectorizer.fit_transform(train_df['text']).toarray()

    n_best = int(len(vectorizer.idf_) * n_best_factor)

    idx_w = np.argsort(vectorizer.idf_)[:n_best]

    train_data = train_data[:, idx_w]
    test_data = vectorizer.transform(test_df['text']).toarray()
    test_data = test_data[:, idx_w]

    # Choose scaler
    max_abs_scaler = preprocessing.MaxAbsScaler()
    # max_abs_scaler = preprocessing.MinMaxScaler()

    ## scale text data for n-gram model ##
    scaled_train_data = max_abs_scaler.fit_transform(train_data)
    scaled_test_data = max_abs_scaler.transform(test_data)

    if use_LSA:
        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components=len(train_data), algorithm='randomized', random_state=43)

        # Word
        scaled_train_data = svd.fit_transform(scaled_train_data)
        scaled_test_data = svd.transform(scaled_test_data)
    return scaled_train_data, scaled_test_data