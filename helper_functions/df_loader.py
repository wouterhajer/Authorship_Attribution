import pandas as pd
import os
from helper_functions.vocabulary import extend_vocabulary
from masking import mask
import csv
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from df_loader_RFM import load_df_RFM


def load_df(args, config):
    if args.corpus_name == 'RFM':
        full_df = load_df_RFM()
    else:
        df_file = args.input_path + os.sep + args.corpus_name + ".csv"
        full_df = pd.read_csv(df_file)

    if args.corpus_name == 'abc_nl1':
        with open('Frequenties.csv', newline='', encoding='MacRoman') as f:
            reader = csv.reader(f)
            vocab_word = list(reader)
            vocab_word = [word[0] for word in vocab_word]
    else:
        vocab_file = args.input_path + os.sep + 'vocab_' + args.corpus_name + ".txt"
        vocab_word = []
        with open(vocab_file, 'r', encoding="utf-8") as fp:
            for line in fp:
                x = line[:-1]
                vocab_word.append(x)

    # If baseline is true a top 100 word-1-gram model is used
    if bool(config['baseline']):
        config['variables']['wordRange'] = [1, 1]
        config['variables']['model'] = "word"
        config['variables']['useLSA'] = 0
        config['masking']['masking'] = 1
        config['masking']['nMasking'] = config['nBaseline']
        config['variables']['nBestFactorWord'] = 1

    #full_df['text'] = [re.sub('[^A-Za-z0-9 ëéèöïüä]+', '', text).lower() for text in full_df['text']]
    #print(full_df.head())

    if args.corpus_name == 'Frida':
        # only keep authors with at least 6 recordings to get a uniform training set
        v = full_df['author'].value_counts()
        full_df = full_df[full_df['author'].isin(v[v >= 6].index)]
        full_df = full_df.reset_index(drop=True)
        full_df = full_df[full_df['text'].map(len) < 5000]
        v = full_df['author'].value_counts()
        full_df = full_df[full_df['author'].isin(v[v >= 8].index)]
        full_df = full_df.reset_index(drop=True)

    # If masking is turned on replace all words outside top n_masking with asterisks
    n_masking = config['masking']['nMasking']
    if bool(config['masking']['masking']):
        config['variables']['nBestFactorWord'] = 1
        #config['variables']['nBestFactorChar'] = 1
        if vocab_word == 0:
            vocab_word = extend_vocabulary([1, 1], full_df['text'], model='word')
        vocab_masking = vocab_word[:n_masking]
        print(vocab_masking)
        vocab_masking = [x.lower() for x in vocab_masking]
        full_df = mask(full_df, vocab_masking, config)

    config['variables']['nAuthors'] = min(config['variables']['nAuthors'], len(list(full_df['author'].unique())))
    print(config['variables']['nAuthors'])

    return full_df, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")

    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    full_df, config = load_df(args, config)
    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])

    # Limit the authors to nAuthors
    author_ids = list(set(full_df['author_id']))
    full_df = full_df.loc[full_df['author_id'].isin(author_ids[:config['variables']['nAuthors']])]
    texts = np.zeros(len(full_df))

    for i, text in enumerate(full_df['text']):
        texts[i] = len(text.split())
    plt.hist(texts, bins=20)
    plt.xlabel('Number of words')
    #plt.title('Number of words in ' + args.corpus_name + ' dataset')
    plt.show()
    print('Mean text length = ' + str(np.mean(texts)))
    print('Median text length = ' + str(np.median(texts)))
    print('Minimal text length = ' + str(np.min(texts)))
    print('Maximal text length = ' + str(np.max(texts)))


if __name__ == '__main__':
    main()
