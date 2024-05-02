import os
import glob
import codecs
import pandas as pd
import json
from split import *
from sklearn.model_selection import train_test_split
from vocabulary import extend_vocabulary
import argparse
from pathlib import Path
import re

def create_df(args, config, p_test = 0.25):
    """
    :param path: Location of the text files that are to be used
    :param config: Dictionary containing globals, see config.json for all variables
    :return: Dataframe containing column 'text' for all texts and 'author' with corresponding labels for the author

    This function is modified per dataset to always output the expected dataframe
    """
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(args.corpus_path + os.sep + '*.txt')
    texts = []

    # keep track of the author of the last document
    author = 0
    if args.corpus_name == "Frida":
        # Hard code conversation numbers for speaker with an odd or even number
        odd = [1, 2, 3, 4, 5, 6, 7, 8]
        even = [3, 4, 1, 2, 7, 8, 5, 6]

        for i, v in enumerate(sorted(files)):
            if v[-5] == '1' and 310 > int(v[6:9]) > 0 and v[9] != 'a':

                # Check if author corresponds to current author and count the conversation
                if author != int(v[6:9]):
                    author = int(v[6:9])
                    j = 0
                else:
                    j += 1

                # Assgin the conversation based on whether the author number is odd or even
                if author % 2 == 1:
                    conversation = odd[j]
                else:
                    conversation = even[j]
                f = codecs.open(v, 'r', encoding='utf-8')
                label = int(v[6:9])
                text = f.read()
                text = text.split('\r\n')
                text2 = []
                for lines in text[1:-1]:
                    line = lines.split('\t')
                    text2.append(line[2])

                text3 = ' '.join(text2[:])

                if len(text3) < 200:
                    print(text3)
                    print(v)
                    continue

                texts.append((text3, label, conversation))
                f.close()
    elif args.corpus_name == "abcnl1":
        for i, v in enumerate(sorted(files)):
            f = codecs.open(v, 'r', encoding='utf-8')
            text = f.read()
            text = text.split('\n\t')
            text2 = []
            for j, line in enumerate(text):
                if j == 0:
                    line = line[1:]
                text2.append(line)
            text3 = ' '.join(text2[:])
            author = int(v[9])
            if v[10] == 'a':
                conv = int(v[11])
            elif v[10] == 'd':
                conv = 3 + int(v[11])
            elif v[10] == 'f':
                conv = 6 + int(v[11])
            #author = (author+conv) % 8 + 1
            #print(author)
            texts.append((text3, author, conv))

    elif args.corpus_name == "abc_nl1":
        for i, v in enumerate(sorted(files)):
            f = codecs.open(v, 'r', encoding=None)
            print(v)
            text = f.read()
            print(text)
            text = text.split('  ')
            text2 = []
            for j, line in enumerate(text):
                print(line)
                if line != '':
                    text2.append(line)

            text3 = ' '.join(text2[:])
            text3 = re.sub(r'\n\n\n', ' ', text3)
            text3 = re.sub(r'\n\n', '\n', text3)
            text3 = re.sub(r'\n', ' ', text3)
            author = int(v[15])
            conv = int(v[18])
            #author = (author+conv) % 8 + 1
            #print(author)
            texts.append((text3, author, conv))
    print(texts)
    # Convert into dataframe
    df = pd.DataFrame(texts, columns=['text', 'author', 'conversation'])
    print(df)
    background_vocab = extend_vocabulary([1, 1], df['text'], model='word')
    vocab_file = args.output_path+os.sep+'vocab_'+args.corpus_name+".txt"

    with open(vocab_file, 'w', encoding="utf-8") as fp:
        fp.write('\n'.join(background_vocab))
    """
    # only keep authors with at least 8 recordings to get a uniform training set
    v = df['author'].value_counts()
    df = df[df['author'].isin(v[v >= 8].index)]
    df = df.reset_index(drop=True)
    """
    print(df)
    df.to_csv(args.output_path+os.sep+args.corpus_name+".csv", index=False)
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=p_test, stratify=df[['author']])
    else:
        print('hello')
        train_df, test_df = split(df, p_test, confusion=bool(config['confusion']))

    return df, train_df, test_df, background_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', help = "Choose the path to the corpus")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help = "Choose the path to output folder")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    df, train_df, test_df, vocab = create_df(args,config)
    print(df[:50])

if __name__ == '__main__':
    main()


