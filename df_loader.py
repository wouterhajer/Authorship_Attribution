import pandas as pd
import os
from vocabulary import extend_vocabulary
from masking import mask
import csv

def load_df(args,config):
    df_file = args.input_path + os.sep + args.corpus_name + ".csv"
    full_df = pd.read_csv(df_file)

    # only keep authors with at least 8 recordings to get a uniform training set
    v = full_df['author'].value_counts()
    full_df = full_df[full_df['author'].isin(v[v >= 6].index)]
    full_df = full_df.reset_index(drop=True)

    vocab_file = args.input_path + os.sep + 'vocab_' + args.corpus_name + ".txt"
    vocab_word = []
    with open(vocab_file, 'r', encoding="utf-8") as fp:
        for line in fp:
            x = line[:-1]
            vocab_word.append(x)
    print(vocab_word[900:1000])

    with open('frequenties.csv', newline='', encoding='MacRoman') as f:
        reader = csv.reader(f)
        print(reader)
        vocab_word = list(reader)
        vocab_word = [word[0] for word in vocab_word]
        print(vocab_word[900:1000])
        #print(vocab_word)

    # If baseline is true a top 100 word-1-gram model is used
    if bool(config['baseline']):
        config['variables']['wordRange'] = [1, 1]
        config['variables']['model'] = "word"
        config['variables']['useLSA'] = 0
        config['masking']['masking'] = 1
        config['masking']['nMasking'] = 100

    # If masking is turned on replace all words outside top n_masking with asterisks
    n_masking = config['masking']['nMasking']
    if bool(config['masking']['masking']):
        #config['variables']['nBestFactorWord'] = 1
        #config['variables']['nBestFactorChar'] = 1
        if vocab_word == 0:
            vocab_word = extend_vocabulary([1, 1], full_df['text'], model='word')
        vocab_masking = vocab_word[:n_masking]
        print(vocab_masking)
        vocab_masking = [x.lower() for x in vocab_masking]
        full_df = mask(full_df, vocab_masking, config)



    return full_df, config
