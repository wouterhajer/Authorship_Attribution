import time
import numpy as np
import json
import random
from sklearn.preprocessing import LabelEncoder
from multiclass_classifier import Multiclass_classifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from df_creator import create_df
from masking import mask
from vocabulary import extend_vocabulary
import pandas as pd
import argparse
import os
from split import split
from sklearn.model_selection import train_test_split

"""
Inspired form Boeninghoff et al  
"""


def ngram(train_df, test_df, config, vocab_word=0):
    """
    :param train_df: Dataframe containing column 'text' for training texts and 'author' with labels for the author
    :param test_df: Dataframe containing column 'text' for test texts and 'author' with labels for the author
    :param config: Dictionary containing globals, see config.json for all variables
    :param background: list of most frequents words in background population if available
    :return: returns lists containing the predicted authors using ensemble, char-ngrams and word-ngrams
    Additionally a list with true authors and a list of booleans corresponding with the confidence of the prediction
    """
    # Shuffle the training data
    #train_df = train_df.sample(frac=1)

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
    print(train_df)
    # If baseline is true a top 100 word-1-gram model is used
    if bool(config['baseline']):
        config['variables']['useLSA'] = 0
        config['variables']['wordRange'] = [1, 1]
        vocab_word = extend_vocabulary([1, 1], train_df['text'], model='word')
        config['variables']['nBestFactorWord'] = 100 / len(vocab_word)

    # Compute predictions using word and character n-gram models (additionaly one focussing on punctuation can be added)
    preds_word, probs_word = Multiclass_classifier(train_df, test_df, config, model='word')
    preds_char, probs_char = Multiclass_classifier(train_df, test_df, config, model='char-std')
    # preds_char_dist, probs_char_dist = Multiclass_classifier(train_df, test_df, config, model='char-dist')

    # Soft Voting procedure (combines the votes of the individual classifier)
    candidates = list(set(train_df['author']))
    test_authors = list(test_df['author'])

    avg_probs = np.average([probs_word, probs_char], axis=0)
    avg_preds = []

    for i, text_probs in enumerate(avg_probs):
        print(i)
        print(text_probs)
        ind_best = np.argmax(text_probs)
        avg_preds.append(candidates[ind_best])

    return avg_preds, preds_char, preds_word, test_authors  # , avg_probs


def test_ngram(args,config):
    start = time.time()

    # Random Seed at file level
    random_seed = 20
    np.random.seed(random_seed)
    random.seed(random_seed)


    df_file = args.input_path + os.sep + args.corpus_name + ".csv"
    full_df = pd.read_csv(df_file)
    # Background vocabulary based on dutch subtitles, lowers performance
    """
    vocab_word = []
    with open('Frequenties.csv', newline='') as csvfile:
        vocab_words = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in vocab_words:

            vocab_word.append(row[0].lower())
            if i > 50000:
                break
            i += 1
    background_vocab = vocab_word[1:]
    """
    vocab_file = args.input_path + os.sep + 'vocab_' + args.corpus_name + ".txt"
    vocab_word = []

    with open(vocab_file, 'r', encoding="utf-8") as fp:
        for line in fp:
            x = line[:-1]
            vocab_word.append(x)
    #full_df, train_df, test_df, vocab_word = create_df('txt', config)
    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    authors = list(set(full_df.author))
    df = full_df.loc[full_df['author'].isin(authors[:config['variables']['nAuthors']])]

    conv = ([2,5,4,7,1,6],[8,3])
    # Use random or deterministic split
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
    else:
        # For now only works without confusion
        train_df, test_df = split(df, 0.25, conversations=conv, confusion=bool(config['confusion']))

    avg_preds, preds_char, preds_word, test_authors = ngram(train_df, test_df, config, vocab_word)

    avg_preds = label_encoder.inverse_transform(avg_preds)
    preds_char = label_encoder.inverse_transform(preds_char)
    preds_word = label_encoder.inverse_transform(preds_word)
    test_authors = label_encoder.inverse_transform(test_authors)

    # Indices where both lists are different
    index = [i for i, x in enumerate(zip(avg_preds, test_authors)) if x[0] != x[1]]
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
    score_partner = 0
    score_rest = 0
    score_sure = 0

    for i in range(len(test_df['author'])):
        if test_authors[i] == avg_preds[i]:
            score += 1
        elif test_authors[i] == avg_preds[i] + 1 and list(test_df['author'])[i] % 2 == 1:
            score_partner += 1
        elif test_authors[i] == avg_preds[i] - 1 and list(test_df['author'])[i] % 2 == 0:
            score_partner += 1
        else:
            score_rest += 1

    print('Score = ' + str(score / len(avg_preds)) + ', random chance = ' + str(1 / len(set(test_authors))))
    print('Score partner = ' + str(score_partner / len(avg_preds)) + ', random chance = ' + str(
        1 / len(set(test_authors))))
    print(
        'Score rest = ' + str(score_rest / len(avg_preds)) + ', random chance = ' + str(1 - 2 / len(set(test_authors))))

    print('Total time: ' + str(time.time() - start) + ' seconds')
    conf = confusion_matrix(test_authors, avg_preds, normalize='true')
    cmd = ConfusionMatrixDisplay(conf, display_labels=set(test_authors))
    cmd.plot()

    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to output folder")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    test_ngram(args, config)



if __name__ == '__main__':
    main()
