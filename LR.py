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

with open('config.json') as f:
    config = json.load(f)

# Random Seed at file level
random_seed = 8
np.random.seed(random_seed)
random.seed(random_seed)

train_df, test_df, vocab_word = create_df('txt', config)

# Encode author labels
label_encoder = LabelEncoder()
train_df['author'] = label_encoder.fit_transform(train_df['author'])
test_df['author'] = label_encoder.transform(test_df['author'])

# Shuffle the training data
# train_df = train_df.sample(frac=1)

# If masking is turned on replace all words outside top n_masking with asterisks
n_masking = config['masking']['nMasking']
if bool(config['masking']['masking']):
    config['masking']['nBestFactorWord'] = 1

    if vocab_word == 0:
        vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
    vocab_word = vocab_word[:n_masking]
    vocab_word = [x.lower() for x in vocab_word]
    train_df = mask(train_df, vocab_word, config)
    test_df = mask(test_df, vocab_word, config)

# If baseline is true a top 100 word-1-gram model is used
if bool(config['baseline']):
    config['variables']['wordRange'] = [1, 1]
    vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
    config['variables']['nBestFactorWord'] = 100 / len(vocab_word)

char_range = tuple(config['variables']['charRange'])
n_best_factor = config['variables']['nBestFactorChar']
lower = bool(config['variables']['lower'])
use_LSA = bool(config['variables']['useLSA'])

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

suspect = 1
train_df['h1'] = [1 if author == suspect else 0 for author in train_df['author']]

h1 = []
h2 = []

for c in conversations:
    char = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
                                                          gamma='auto')))
    train = train_df.index[train_df['conversation'] != c].tolist()

    calibrate = train_df.index[train_df['conversation'] == c].tolist()


    char.fit(scaled_train_data_word[train], train_df['h1'][train])

    probas_char = char.predict_proba(scaled_train_data_word[calibrate])
    print(train_df['h1'][calibrate])
    print(probas_char)
    h1.extend([probas_char[i, 0] for i in range(len(probas_char)) if list(train_df['h1'][calibrate])[i] == 1])
    h2.extend([probas_char[i, 0] for i in range(len(probas_char)) if list(train_df['h1'][calibrate])[i] == 0])
print(h1)
print(h2)
print(test_df['author'])
#print(probas_char)

# New stuff
bins=np.histogram(np.hstack((h1,h2)), bins=20)[1] #get the bin edges

plt.hist(h1,bins,density=True)
plt.hist(h2,bins,density=True)
plt.show()