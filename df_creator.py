import os
import glob
import codecs
import pandas as pd
import json
from split import *
from sklearn.model_selection import train_test_split
from vocabulary import extend_vocabulary
def create_df(path, config, p_test = 0.25):
    """
    :param path: Location of the text files that are to be used
    :param config: Dictionary containing globals, see config.json for all variables
    :return: Dataframe containing column 'text' for all texts and 'author' with corresponding labels for the author

    This function is modified per dataset to always output the expected dataframe
    """
    # Reads all text files located in the 'path' and assigns them to 'label' class
    n_words = config["variables"]["nWords"]

    files = glob.glob(path + os.sep + '*.txt')

    texts = []

    # keep track of the author of the last document
    author = 0

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

            if len(text3)<200:
                print(text3)
                print(v)
                continue

            # If trueLength is set to 1, cut off messages at that amount of words and get rid of shorter messages
            # Not in active use, depreciated
            if bool(config["variables"]["trueLength"]) and not bool(config["variables"]["concatenate"]):
                # Makes the messages all of the same amount of words (Nwords in the config file)

                words = text3.split(' ')

                # Removing these markers drastically lowers performance!
                """
                words2= []
                for word in words:
                    if word != 'ggg' and word != 'xxx':
                        words2.append(word)
                words = words2
                """

                if len(words) < 100:
                    print(v)
                    f.close()
                    continue
                if len(words) >= n_words:
                    text4 = ' '.join(words[:n_words])
                    texts.append((text4, label, conversation))
                if len(words) >= 2 * n_words:
                    text4 = ' '.join(words[-n_words:])
                    texts.append((text4, label, conversation))
            else:
                if len(text3) > 10:
                    texts.append((text3, label, conversation))
            f.close()

    # Convert into dataframe
    df = pd.DataFrame(texts, columns=['text', 'author', 'conversation'])



    # Concatenate all texts of a single author and split in messages of even length
    # Not in active use, depreciated
    if config["variables"]["concatenate"]:
        new_texts = []
        for author in set(list(df['author'])):
            text = df.loc[df["author"] == author]
            conc = ' '.join(list(text['text']))
            words = conc.split(' ')
            for i in range(len(words) // n_words):
                sentence = words[i * n_words:(i + 1) * n_words]
                new_texts.append((' '.join(sentence[:]), author))
        df = pd.DataFrame(new_texts, columns=['text', 'author'])

    background_vocab = extend_vocabulary([1, 1], df['text'], model='word')

    # only keep authors with at least 8 recordings to get a uniform training set
    v = df['author'].value_counts()
    df = df[df['author'].isin(v[v >= 8].index)]
    df = df.reset_index(drop=True)

    # Limit the authors to nAuthors
    df = df.loc[df['author'] < config['variables']['nAuthors']]
    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(df, test_size=p_test, stratify=df[['author']])
    else:
        print('hello')
        train_df, test_df = split(df, p_test, confusion=bool(config['confusion']))

    return df, train_df, test_df, background_vocab


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    df, train_df, test_df, vocab = create_df('txt', config)
    print(train_df[:50])


