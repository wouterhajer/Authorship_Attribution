import pandas as pd
import os
from helper_functions.masking import mask
import csv
from df_loader_RFM import load_df_RFM


def load_df(args, config):
    # Load dataframe from file
    if args.corpus_name == 'RFM':
        full_df = load_df_RFM()
    else:
        df_file = args.input_path + os.sep + args.corpus_name + ".csv"
        full_df = pd.read_csv(df_file)

    # Use SUBTLEX_NL frequency list for abc_nl, use those created with df_creator2.py for other corpora
    if args.corpus_name == 'abc_nl1':
        with open('../Frequenties.csv', newline='', encoding='MacRoman') as f:
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

    if args.corpus_name == 'Frida':
        full_df = full_df[full_df['text'].map(len) < 5000]
        v = full_df['author'].value_counts()
        # only keep authors with the full 8 recordings to get a uniform training set
        full_df = full_df[full_df['author'].isin(v[v >= 8].index)]
        full_df = full_df.reset_index(drop=True)

    # If masking is turned on replace all words outside top n_masking with hashtags
    n_masking = config['masking']['nMasking']
    if bool(config['masking']['masking']):
        config['variables']['nBestFactorWord'] = 1
        #config['variables']['nBestFactorChar'] = 1
        vocab_masking = vocab_word[:n_masking]
        vocab_masking = [x.lower() for x in vocab_masking]
        full_df = mask(full_df, vocab_masking, config)

    config['variables']['nAuthors'] = min(config['variables']['nAuthors'], len(list(full_df['author'].unique())))
    return full_df, config
