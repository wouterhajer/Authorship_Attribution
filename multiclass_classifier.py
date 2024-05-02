from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from data_scaler import data_scaler,data_scaler_embedding
import json
import numpy as np
import ast

def Multiclass_classifier(train_df, test_df, config, model):
    """
    Trains classifier on training set and predict classifications on test_df
    @param train_df: dataframe containing the training texts and true authors
    @param test_df: dataframe containing the test texts and true authors
    @param config: config file with parameters
    @param model: Specify if char-std, word or char-dist model is to be used
    @return: list of predicted author and matrix of probabilities per text
    """

    #train = np.array([np.fromstring(embedding[1:-1], dtype=float, sep=' ') for embedding in train_df['embedding']]) #np.array([ast.literal_eval(embedding) for embedding in train_df['embedding']])
    #test = np.array([np.fromstring(embedding[1:-1], dtype=float, sep=' ') for embedding in test_df['embedding']])#np.array([ast.literal_eval(embedding) for embedding in test_df['embedding']])
    #scaled_train_data, scaled_test_data = data_scaler_embedding(train, test)
    scaled_train_data, scaled_test_data = data_scaler(train_df, test_df, config, model=model)
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',gamma='auto')))
    char.fit(scaled_train_data, train_df['author_id'])
    preds_char = char.predict(scaled_test_data)
    probas_char = char.predict_proba(scaled_test_data)
    return preds_char, probas_char


def binary_classifier(train_data, truth, test_data):
    """
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear')))
    char.fit(train_data, truth)
    probas_char = char.predict_proba(test_data)
    """
    char2 = SVC(kernel='linear', C=1)
    char2.fit(train_data, truth)
    decision_values = char2.decision_function(test_data)

    return -decision_values # probas_char[:,0] #
