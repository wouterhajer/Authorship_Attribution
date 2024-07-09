from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from vocabulary import extend_vocabulary
import numpy as np

# Function modified from the code created by Lukas Muttenthaler, Gordon Lucas and Janek Amann
# The original code can be found at https://github.com/pan-webis-de/muttenthaler19

def data_scaler(train_df,test_df,config,model = 'char-std'):
    """
    :param train_df: Training dataframe
    :param test_df: Validation dataframe
    :param config: Fixed variables from json file
    :param model: Model type, either 'char-std' or 'word'
    :return: Two arrays containing training and validation feature vectors of all texts

    Creates feature vectors from the texts in both dataframes.
    These vectors are then modified and scaled before returning them.
    """
    # Specify variables based on specified model
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

    # Find all unique features in the training set
    vocab = extend_vocabulary(range, train_df['text'], model=model)

    # Count the occurrences of all features (Tfidf scaling is later fixed with rescaling)
    vectorizer = TfidfVectorizer(analyzer=model[:4], ngram_range=range, use_idf=True,
                                      norm='l2', lowercase=lower, vocabulary=vocab,
                                      smooth_idf=True, sublinear_tf=True)

    # Count all features on training and validation set
    train_data = vectorizer.fit_transform(train_df['text']).toarray()
    test_data = vectorizer.transform(test_df['text']).toarray()

    # Restrict to only the 30% most frequent features
    n_best = int(len(vectorizer.idf_) * n_best_factor)
    idx_w = np.argsort(vectorizer.idf_)[:n_best]
    train_data = train_data[:, idx_w]
    test_data = test_data[:, idx_w]

    # Scale the feature vectors
    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data = max_abs_scaler.fit_transform(train_data)
    scaled_test_data = max_abs_scaler.transform(test_data)

    if use_LSA:
        # initialize truncated singular value decomposition to shorten feature vectors
        svd = TruncatedSVD(n_components=len(train_data), algorithm='randomized', random_state=43)
        scaled_train_data = svd.fit_transform(scaled_train_data)
        scaled_test_data = svd.transform(scaled_test_data)

    return scaled_train_data, scaled_test_data
