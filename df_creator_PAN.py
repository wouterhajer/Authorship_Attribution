import os
import glob
import codecs
import pandas as pd
import json


def read_files(path: str, label: str):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = sorted(glob.glob(path + os.sep + label + os.sep + '*.txt'))
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        f.close()
    return texts


def create_df_PAN(path, problem):
    problem = 'problem00001'
    # Reading information about the problem
    infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
    candidates = []
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        unk_folder = fj['unknown-folder']
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    # building training set
    train_docs = []
    for candidate in candidates:
        train_docs.extend(read_files(path + os.sep + problem, candidate)[:])

    train_df = pd.DataFrame(train_docs, columns=['text', 'author'])

    # building test set
    test_docs = read_files(path + os.sep + problem, unk_folder)
    test_texts = [text[:] for (text, label) in test_docs]

    # Make list of which test texts the author is known
    with open(path + os.sep + problem + os.sep + 'ground-truth.json') as f:
        truth = json.load(f)

    truth_list = []
    known = []
    for i, j in enumerate(truth['ground_truth']):
        # Check if authorname is not <unknown>
        if j['true-author'][-1] != '>':
            truth_list.append(j['true-author'])
            known.append(i)
        else:
            truth_list.append(-1)
    # Clean the test texts by removing the ones with unknown author
    known_authors = [truth_list[x] for x in known]
    test_texts = [test_texts[x] for x in known]
    test_df = pd.DataFrame({'text': test_texts, 'author': known_authors})
    return train_df, test_df


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    print(create_df_PAN('Pan2019', problem='problem00001'))
