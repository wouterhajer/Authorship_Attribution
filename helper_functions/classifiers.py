from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from helper_functions.data_scaler import data_scaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def Multiclass_classifier(train_df, test_df, config, model):
    """
    Trains classifier on training set and predict classifications on test_df
    :param train_df: dataframe containing the training texts and true authors
    :param test_df: dataframe containing the test texts and true authors
    :param config: dictionary containing globals, see config.json for all variables
    :param model: specify if char-std or word model is to be used
    :return: list of predicted author and matrix of probabilities per text
    """

    # Calculate feature vectors
    scaled_train_data, scaled_test_data = data_scaler(train_df, test_df, config, model=model)

    # Initialize classifiers
    if config['baseline'] == 1:
        char = CalibratedClassifierCV(OneVsRestClassifier(LogisticRegression(C=1)))
    else:
        char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear', gamma='auto')))

    #Fit classifier and compute results
    char.fit(scaled_train_data, train_df['author_id'])
    preds_char = char.predict(scaled_test_data)
    probas_char = char.predict_proba(scaled_test_data)
    return preds_char, probas_char


def binary_classifier(train_data, truth, test_data, config):
    """
   Trains binary classifier on training set and predict classifications on test_df
   :param train_data: feature vectors of training data
   :param truth: truth value of training data
   :param test_data: feature vectors of validation data
   :param config: dictionary containing globals, see config.json for all variables
   :return: array of decision values for all test feature vectors
   """
    if config['baseline'] == 1:
        char = LogisticRegression(C=1)
    else:
        char = SVC(kernel='linear', C=1)

    char.fit(train_data, truth)
    decision_values = char.decision_function(test_data)

    return -decision_values

def feature_based_classification(train_df, test_df, config):
    """
    :param train_df: Dataframe containing column 'text' for training texts and 'author' with labels for the author
    :param test_df: Dataframe containing column 'text' for test texts and 'author' with labels for the author
    :param config: Dictionary containing globals, see config.json for all variables
    :return: returns lists containing the predicted authors using the specified model and the true authors
    """

    candidates = list(set(train_df['author_id']))
    test_authors = list(test_df['author_id'])

    avg_preds = []

    # Compute predictions using word and character n-gram models
    if config['variables']['model'] == 'char':
        preds_char, probs_char = Multiclass_classifier(train_df, test_df, config, model='char-std')
        avg_preds = preds_char
    elif config['variables']['model'] == 'word':
        preds_word, probs_word = Multiclass_classifier(train_df, test_df, config, model='word')
        avg_preds = preds_word
    elif config['variables']['model'] == 'both':
        preds_char, probs_char = Multiclass_classifier(train_df, test_df, config, model='char-std')
        preds_word, probs_word = Multiclass_classifier(train_df, test_df, config, model='word')
        avg_probs = np.average([probs_word, probs_char], axis=0)
        for i, text_probs in enumerate(avg_probs):
            ind_best = np.argmax(text_probs)
            avg_preds.append(candidates[ind_best])

    return avg_preds, test_authors


def SVM_model_scores(train_word, truth_word, test_word, train_char, truth_char, test_char, model, config):
    """
   :param train_word/train_char: training feature vectors for that feature
   :param truth_word/truth_char: truth values for that feature
   :param test_word/test_char: test feature vectors for that feature
   :param model: specify if char or word model is to be used
   :param config: Dictionary containing globals, see config.json for all variables
   :return: array of decision values for all test feature vectors
   """
    if model == 'both':
        word = binary_classifier(train_word, truth_word, test_word, config)
        char = binary_classifier(train_char, truth_char, test_char, config)
        return (word + char) / 2
    elif model == 'word':
        return binary_classifier(train_word, truth_word, test_word, config)
    elif model == 'char':
        return binary_classifier(train_char, truth_char, test_char, config)
