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
from df_creator_PAN import *
from df_creator import *

# Assign device and check if it is GPU or not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Download new bertmodel from huggingface and saves locally
''' 
model = BertModel.from_pretrained('bert-base-cased')
model.save_pretrained("BERTmodels/bert-base-cased")
'''

# Get path to directory and specify problem
path = 'Pan2019'
problem = 'problem00001'

with open('config.json') as f:
    config = json.load(f)

n_authors = config['variables']['nAuthors']

train_df, test_df = create_df_PAN(path, problem)
print(train_df)
full_df, train_df, test_df, background_vocab = create_df('txt', config) #create_df_PAN(path, problem)
print(train_df)

# Encode author labels
label_encoder = LabelEncoder()
train_df['author_id'] = label_encoder.fit_transform(train_df['author'])
# Limit the authors to nAuthors
authors = list(set(full_df.author))
train_df = train_df.loc[train_df['author'].isin(authors[:n_authors])]
test_df = test_df.loc[test_df['author'].isin(authors[:n_authors])]

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



# Set tokenizer and tokenize training and test texts
# tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')  #
train_encodings = transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256, \
                                    device=device)
val_encodings = transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256, \
                                        device=device)
print(set(train_df['author']))
print(set(test_df['author']))

encoded_known_authors = label_encoder.transform(test_df['author'])
train_labels = torch.tensor(train_df['author_id'], dtype=torch.long).to(device)
N_classes = len(list(set(encoded_known_authors)))
print(N_classes)
# Define the model for fine-tuning
#bert_model = RobertaModel.from_pretrained('BERTmodels/robbert-2023-dutch-base')
bert_model = BertModel.from_pretrained('BERTmodels/bert-base-dutch-cased')
model = BertMeanPoolingClassifier(bert_model, N_classes=N_classes, dropout=config['BERT']['dropout'])
model.to(device)

# Set up DataLoader for training
dataset = CustomDataset(train_encodings, train_labels)
batch_size = 1
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning and validation loop
epochs = 5

for j in range(8):
    model = finetune_bert_meanpooling(model, train_dataloader, epochs, config)

    print('validation set')
    preds, scores = validate_bert_meanpooling(model, val_encodings, encoded_known_authors)
    avg_preds = label_encoder.inverse_transform(preds)
    author_number = [author for author in test_df['author']]
    conf = confusion_matrix(test_df['author'], avg_preds, normalize='true')
    cmd = ConfusionMatrixDisplay(conf, display_labels=sorted(set(author_number)))
    cmd.plot()
    plt.show()

