import time
import numpy as np
import json
import random
from sklearn.preprocessing import LabelEncoder
from helper_functions.multiclass_classifier import Multiclass_classifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
from helper_functions.split import split
from sklearn.model_selection import train_test_split
from helper_functions.df_loader import load_df
from df_loader_RFM import load_df_RFM


def ngram(train_df, test_df, config):
    """
    :param train_df: Dataframe containing column 'text' for training texts and 'author' with labels for the author
    :param test_df: Dataframe containing column 'text' for test texts and 'author' with labels for the author
    :param config: Dictionary containing globals, see config.json for all variables
    :param background: list of most frequents words in background population if available
    :return: returns lists containing the predicted authors using ensemble, char-ngrams and word-ngrams
    Additionally a list with true authors and a list of booleans corresponding with the confidence of the prediction
    """

    candidates = list(set(train_df['author_id']))
    test_authors = list(test_df['author_id'])

    avg_preds = []

    # Compute predictions using word and character n-gram models (additionaly one focussing on punctuation can be added)
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


def test_ngram(args,config):
    start = time.time()

    # Random Seed at file level
    random_seed = 20
    np.random.seed(random_seed)
    random.seed(random_seed)
    if args.corpus_name == 'RFM':
        full_df, config = load_df_RFM(args, config)
    else:
        full_df, config = load_df(args,config)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    author_ids = list(set(full_df['author_id']))
    df = full_df.loc[full_df['author_id'].isin(author_ids[:config['variables']['nAuthors']])]

    #conv = ([2,5,4,3,6],[1])
    print(df)
    # Use random or deterministic split
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
    else:
        # For now only works without confusion
        train_df, test_df = split(df, 0.25, confusion=bool(config['confusion']))

    avg_preds, test_authors = ngram(train_df, test_df, config)

    avg_preds = label_encoder.inverse_transform(avg_preds)
    test_authors = label_encoder.inverse_transform(test_authors)

    # Indices where both lists are different
    index = [i for i, x in enumerate(zip(avg_preds, test_authors)) if x[0] != x[1]]
    print([avg_preds[x] for x in index])
    print([test_authors[x] for x in index])
    print('Included authors: ' + str(len(set(test_authors))))

    # Calculate F1 score
    f1 = f1_score(test_authors, avg_preds, average='macro')
    print('F1 Score:' + str(f1))

    score = 0
    score_partner = 0
    score_rest = 0

    # Calculate the scores
    if args.corpus_name == 'Frida' or args.corpus_name == 'RFM':
        for j in range(len(test_df['author'])):
            if test_authors[j] == avg_preds[j]:
                score += 1
            elif test_authors[j] == avg_preds[j] - 1 and test_authors[j] % 2 == 1:
                score_partner += 1
            elif test_authors[j] == avg_preds[j] + 1 and test_authors[j] % 2 == 0:
                score_partner += 1
            else:
                score_rest += 1
    # Calculate the scores
    elif args.corpus_name == 'abc_nl1':
        for j in range(len(test_df['author'])):
            if test_authors[j] == avg_preds[j]:
                score += 1
            elif avg_preds[j] % 2 ==0 and test_authors[j] % 2 == 1:
                score_partner += 1
            elif avg_preds[j] % 2 == 1 and test_authors[j] % 2 == 0:
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
