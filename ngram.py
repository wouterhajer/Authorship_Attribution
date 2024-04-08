import time
import numpy as np
import json
import random
from sklearn.preprocessing import LabelEncoder
from multiclass_classifier import Multiclass_classifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
from split import split
from sklearn.model_selection import train_test_split
from df_loader import load_df

"""
Modified from Boeninghoff et al  
"""

def ngram(train_df, test_df, config):
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

    # Compute predictions using word and character n-gram models (additionaly one focussing on punctuation can be added)
    preds_word, probs_word = Multiclass_classifier(train_df, test_df, config, model='word')
    preds_char, probs_char = Multiclass_classifier(train_df, test_df, config, model='char-std')
    # preds_char_dist, probs_char_dist = Multiclass_classifier(train_df, test_df, config, model='char-dist')

    # Soft Voting procedure (combines the votes of the individual classifier)
    candidates = list(set(train_df['author_id']))
    test_authors = list(test_df['author_id'])

    avg_probs = np.average([probs_word, probs_char], axis=0)
    avg_preds = []

    for i, text_probs in enumerate(avg_probs):
        ind_best = np.argmax(text_probs)
        avg_preds.append(candidates[ind_best])

    return avg_preds, preds_char, preds_word, test_authors  # , avg_probs


def test_ngram(args,config):
    start = time.time()

    # Random Seed at file level
    random_seed = 20
    np.random.seed(random_seed)
    random.seed(random_seed)

    full_df,config = load_df(args,config)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    author_ids = list(set(full_df['author_id']))
    df = full_df.loc[full_df['author_id'].isin(author_ids[:config['variables']['nAuthors']])]

    conv = ([2,5,4,7,1,6],[8,3])
    # Use random or deterministic split
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
    else:
        # For now only works without confusion
        train_df, test_df = split(df, 0.25, conversations=conv, confusion=bool(config['confusion']))

    avg_preds, preds_char, preds_word, test_authors = ngram(train_df, test_df, config)

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

    # Calculate the scores
    for j in range(len(test_df['author'])):
        if test_authors[j] == avg_preds[j]:
            score += 1
        elif test_authors[j] == avg_preds[j] - 1 and test_authors[j] % 2 == 1:
            score_partner += 1
        elif test_authors[j] == avg_preds[j] + 1 and test_authors[j] % 2 == 0:
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
