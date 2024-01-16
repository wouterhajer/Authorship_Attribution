import random
import json
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from word import *
from char import *
from char_dist import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from df_creator import *
import seaborn as sn

if __name__ == '__main__':
    start = time.time()
    with open('config.json') as f:
        config = json.load(f)
    # Random Seed at file level
    random_seed = 38
    np.random.seed(random_seed)
    random.seed(random_seed)

    df = read_files('txt', config)
    print(df)
    # Encode author labels
    label_encoder = LabelEncoder()
    df['author'] = label_encoder.fit_transform(df['author'])

    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])

    # shuffle training data
    train_df = train_df.sample(frac=1)
    n_masking = config['masking']['nMasking']
    if bool(config['masking']['masking']):
        n_best_factor = 1
        vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
        vocab_word = vocab_word[:n_masking]

        train_df = mask(train_df, vocab_word, config)
        test_df = mask(test_df, vocab_word, config)

    print('Start SVMs after ' + str(time.time() - start) + ' seconds')
    preds_word, probs_word = word_gram(train_df, test_df, config)
    print('End word SVM after ' + str(time.time() - start) + ' seconds')
    preds_char, probs_char = char_gram(train_df, test_df, config)
    # preds_char_dist, probs_char_dist = char_dist_gram(train_df, test_df, config)

    print('End SVMs after ' + str(time.time() - start) + ' seconds\n')

    # Soft Voting procedure (combines the votes of the three individual classifier)
    candidates = list(set(df['author']))
    n_authors = (len(candidates))
    test_authors = list(test_df['author'])

    avg_probs = np.average([probs_word, probs_char], axis=0)
    avg_preds = []
    diff_probs = []
    sure = []

    for i, text_probs in enumerate(avg_probs):
        # plt.scatter(candidates, text_probs)
        # plt.scatter(test_authors[i], text_probs[test_authors[i]], color='r')
        # plt.show()
        ind_best = np.argmax(text_probs)
        avg_preds.append(candidates[ind_best])
        probs_copy = text_probs.copy()
        true_prob = text_probs[test_authors[i]]
        max_false_prob = max(np.delete(probs_copy, test_authors[i]))
        diff_probs.append(true_prob - max_false_prob)

        second = np.partition(text_probs, -2)[-2]
        if text_probs.max() - second > 1 / n_authors:
            sure.append(True)
        else:
            sure.append(False)
    avg_preds = label_encoder.inverse_transform(avg_preds)

    preds_char = label_encoder.inverse_transform(preds_char)
    preds_word = label_encoder.inverse_transform(preds_word)
    test_authors = label_encoder.inverse_transform(test_authors)
    # Indices where both lists are different
    index = [i for i, x in enumerate(zip(avg_preds,test_authors)) if x[0] != x[1]]
    print([avg_preds[x] for x in index])
    print([test_authors[x] for x in index])
    print('Included authors: ' + str(len(set(test_authors))))

    # Calculate F1 score
    f1 = f1_score(test_authors, avg_preds, average='macro')
    print('F1 Score average:' + str(f1))

    f1 = f1_score(test_authors, preds_char, average='macro')
    print('F1 Score char:' + str(f1))

    # f1 = f1_score(test_authors, preds_char_dist, average='macro')
    # print('F1 Score char dist:' + str(f1))

    f1 = f1_score(test_authors, preds_word, average='macro')
    print('F1 Score word:' + str(f1) + '\n')

    score = 0
    score_sure = 0

    for i in range(len(test_df['author'])):
        if test_authors[i] == avg_preds[i]:
            score += 1
            if sure[i]:
                score_sure += 1

    print('Percentage sure: ' + str(sure.count(True) / len(sure)))
    print('Score when sure: ' + str(score_sure / sure.count(True)))
    indeces = [i for i, x in enumerate(sure) if x]
    f1 = f1_score([test_authors[x] for x in indeces], [avg_preds[x] for x in indeces], average='macro')
    print('Macro F1 when sure:' + str(f1))
    indeces = [i for i, x in enumerate(sure) if not x]
    f1 = f1_score([test_authors[x] for x in indeces], [avg_preds[x] for x in indeces], average='macro')
    print('Macro F1 when unsure:' + str(f1) + '\n')

    """
    print(sum(diff_probs) / len(diff_probs))
    # Get all positive numbers into another list
    pos_only = [x for x in diff_probs if x > 0]
    if pos_only:
        print(sum(pos_only) / len(pos_only))

    neg_only = [x for x in diff_probs if x < 0]
    if neg_only:
        print(sum(neg_only) / len(neg_only))
    """

    print('Total time: ' + str(time.time() - start) + ' seconds')
    conf = confusion_matrix(test_authors, avg_preds, normalize='true')
    cmd = ConfusionMatrixDisplay(conf, display_labels=set(test_authors))
    cmd.plot()

    plt.show()
