import os
import glob
import codecs
import pandas as pd
import json


def read_files(path, config):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    n_words = config["variables"]["nWords"]

    files = glob.glob(path + os.sep + '*.txt')
    texts = []
    tot = 0
    for i, v in enumerate(files):

        if v[-5] == '1' and 31 > int(v[6:9]) > 0:  # and int(v[6:9]) % 2 == 1:
            f = codecs.open(v, 'r', encoding='utf-8')
            label = int(v[6:9])
            text = f.read()
            text = text.split('\r\n')
            text2 = []
            for j in text[1:-1]:
                line = j.split('\t')
                text2.append(line[2])

            text3 = ' '.join(text2[:])

            if bool(config["variables"]["trueLength"]) and not bool(config["variables"]["concatenate"]):
                # Makes the messages all of the same amount of words (Nwords in the config file)
                # Removing these markers drastically lowers performance!
                """
                words2= []
                for word in words:
                    if word != 'ggg' and word != 'xxx':
                        words2.append(word)
                words = words2
                """
                words = text3.split(' ')
                tot += len(words)
                if len(words) < 100:
                    print(v)
                    f.close()
                    continue
                if len(words) >= n_words:
                    text4 = ' '.join(words[:n_words])
                    texts.append((text4, label))
                if len(words) >= 2*n_words:
                    text4 = ' '.join(words[n_words:n_words*2])
                    texts.append((text4, label))
            else:
                texts.append((text3, label))

            f.close()


    df = pd.DataFrame(texts, columns=['text', 'author'])

    new_texts = []
    if config["variables"]["concatenate"]:
        for author in set(list(df['author'])):
            text = df.loc[df["author"] == author]
            conc = ' '.join(list(text['text']))
            words = conc.split(' ')
            for i in range(len(words)//n_words):
                sentence = words[i*n_words:(i+1)*n_words]
                new_texts.append((' '.join(sentence[:]), author))

        df = pd.DataFrame(new_texts, columns=['text', 'author'])
    # only keep authors with at least 7 recordings.
    v = df['author'].value_counts()
    df = df[df['author'].isin(v[v >= 8].index)]
    df = df.reset_index(drop=True)

    return df

if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    print(read_files('txt',config)[:50])