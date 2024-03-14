from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from data_scaler import data_scaler

def char_gram(train_df, test_df, config):
    scaled_train_data_char, scaled_test_data_char = data_scaler(train_df,test_df,config,model = 'char-std')
    preds_char, probas_char = SVM_classifier2(scaled_train_data_char, scaled_test_data_char, train_df)
    return preds_char,probas_char

def SVM_classifier2(scaled_train_data, scaled_test_data, train_df):
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    char.fit(scaled_train_data, train_df['author'])
    preds_char = char.predict(scaled_test_data)
    probas_char = char.predict_proba(scaled_test_data)
    return preds_char, probas_char

def SVM_classifier(train_df, test_df, config,model):
    scaled_train_data, scaled_test_data = data_scaler(train_df, test_df, config, model=model)
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    char.fit(scaled_train_data, train_df['author'])
    preds_char = char.predict(scaled_test_data)
    probas_char = char.predict_proba(scaled_test_data)
    return preds_char, probas_char
