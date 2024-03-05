import torch
from torch import Tensor
from torch.nn import Module
import gc
import pandas as pd
import os
import glob
import codecs
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay, confusion_matrix
from transformers import BatchEncoding, PreTrainedTokenizerBase
import time
from text_transformer import *
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Optional, Union
from torch.nn import BCELoss, DataParallel, Module, Linear, Sigmoid
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase, RobertaModel
import csv
import torch
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from masking import *
from BERT_meanpooling import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
''' 
model = BertModel.from_pretrained('bert-base-cased')
model.save_pretrained("BERTmodels/bert-base-cased")
'''


def read_files(path: str, label: str):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = sorted(glob.glob(path + os.sep + label + os.sep + '*.txt'))
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        f.close()
    return texts


with open('config.json') as f:
    config = json.load(f)

# get directory and specify problem
path = 'Pan2019'
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
    train_docs.extend(read_files(path + os.sep + problem, candidate))

# Convert to dataframe
train_df = pd.DataFrame(train_docs, columns=['text', 'author'])

# Shuffle training data
train_df = train_df.sample(frac=1)

# Building test set
test_docs = read_files(path + os.sep + problem, unk_folder)
test_texts = [text for (text, label) in test_docs]

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
test = [(test_texts[i], known_authors[i]) for i in range(len(test_texts))]
test_df = pd.DataFrame(test, columns=['text', 'author'])

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # RobertaTokenizer.from_pretrained('roberta-base') #

if bool(config['masking']['masking']):
    vocab_word = []
    with open(path + os.sep + '5000English.csv', newline='') as csvfile:
        vocab_words = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in vocab_words:
            vocab_word.append(row[0].lower())
    print(vocab_word[:100])
    vocab_word = vocab_word[1:config['masking']['nMasking'] + 1]
    train_df = mask(train_df, vocab_word, config)
    print(train_df['text'][0])
    test_df = mask(test_df, vocab_word, config)

encodings = transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256, \
                                    device=device)
val_encodings = transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256, \
                                        device=device)
# Encode author labels
label_encoder = LabelEncoder()
train_df['author_id'] = label_encoder.fit_transform(train_df['author'])
encoded_known_authors = label_encoder.transform(known_authors)
train_labels = torch.tensor(train_df['author_id'], dtype=torch.long).to(device)
print(encodings)
# Define the model for fine-tuning
model = BertMeanPoolingClassifier(N_classes=9, dropout=config['BERT']['dropout'])
model.to(device)

test = model(encodings[0])
print(test)
dataset = CustomDataset(encodings, train_labels)
# Set up DataLoader for training
batch_size = 1
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=config['BERT']['learningRate'])
criterion = torch.nn.CrossEntropyLoss()

# Fine-tuning loop
epochs = config['BERT']['epochs']

for epoch in range(epochs):
    model.train()
    total_loss = 0
    i = 0
    for batch in train_dataloader:
        encoding, labels = batch['encodings'], batch['labels'][0]

        encoding = {'input_ids': encoding['input_ids'][0], \
                    'token_type_ids': encoding['token_type_ids'][0], \
                    'attention_mask': encoding['attention_mask'][0]
                    }
        # optimizer.zero_grad()
        outputs = model(encoding)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        if i % 9 == 8:
            optimizer.step()
            optimizer.zero_grad()
        i += 1

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")
    if epoch % 2 == 1:
        print('validation set')
        preds = validate(model, val_encodings, encoded_known_authors)
        avg_preds = label_encoder.inverse_transform(preds)
        author_number = [author[-2:] for author in known_authors]
        conf = confusion_matrix(known_authors, avg_preds, normalize='true')
        cmd = ConfusionMatrixDisplay(conf, display_labels=sorted(set(author_number)))
        cmd.plot()
        plt.show()
    # delete locals
    del encoding
    del outputs
    del loss
    # Then clean the cache
    torch.cuda.empty_cache()
    # then collect the garbage
    gc.collect()
