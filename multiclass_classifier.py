from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from data_scaler import data_scaler


def Multiclass_classifier(train_df, test_df, config, model):
    """
    Trains classifier on training set and predict classifications on test_df
    @param train_df: dataframe containing the training texts and true authors
    @param test_df: dataframe containing the test texts and true authors
    @param config: config file with parameters
    @param model: Specify if char-std, word or char-dist model is to be used
    @return: list of predicted author and matrix of probabilities per text
    """
    scaled_train_data, scaled_test_data = data_scaler(train_df, test_df, config, model=model)
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    char.fit(scaled_train_data, train_df['author'])
    preds_char = char.predict(scaled_test_data)
    probas_char = char.predict_proba(scaled_test_data)
    return preds_char, probas_char

def binary_classifier(train_data, truth, test_data):
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear', gamma='auto')))
    char.fit(train_data, truth)
    probas_char = char.predict_proba(test_data)
    return probas_char[:,0]
