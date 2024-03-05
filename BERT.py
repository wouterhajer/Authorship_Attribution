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
from df_creator_PAN import *
from df_creator import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
''' 
model = BertModel.from_pretrained('bert-base-cased')
model.save_pretrained("BERTmodels/bert-base-cased")
'''
# get directory and specify problem
path = 'Pan2019'
problem = 'problem00001'

with open('config.json') as f:
    config = json.load(f)

train_df, test_df = create_df_PAN(path,problem)

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

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # RobertaTokenizer.from_pretrained('roberta-base') #
encodings = transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256, \
                                    device=device)
val_encodings = transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256, \
                                        device=device)

# Encode author labels
label_encoder = LabelEncoder()
train_df['author_id'] = label_encoder.fit_transform(train_df['author'])
encoded_known_authors = label_encoder.transform(test_df['author'])
train_labels = torch.tensor(train_df['author_id'], dtype=torch.long).to(device)

# Define the model for fine-tuning
model = BertMeanPoolingClassifier(N_classes=9, dropout=config['BERT']['dropout'])
model.to(device)

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
    if epoch % 5 == 4:
        print('validation set')
        preds = validate(model, val_encodings, encoded_known_authors)
        avg_preds = label_encoder.inverse_transform(preds)
        author_number = [author[-2:] for author in test_df['author']]
        conf = confusion_matrix(test_df['author'], avg_preds, normalize='true')
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
