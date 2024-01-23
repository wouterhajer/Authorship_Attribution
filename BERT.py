import random
import json
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from df_creator import read_files

with open('config.json') as f:
    config = json.load(f)

# Random Seed at file level
random_seed = 43
np.random.seed(random_seed)
random.seed(random_seed)

df = read_files('txt',config)

#only keep authors with at least 8 recordings.
v = df['author'].value_counts()
df = df[df['author'].isin(v[v == 8].index)]

train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])

# Encode author labels
label_encoder = LabelEncoder()
train_df['author_id'] = label_encoder.fit_transform(train_df['author'])

# Tokenize and encode the training data using Dutch BERT
tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
X_train = tokenizer(list(train_df['text']), padding=True, truncation=True, return_tensors='pt', max_length=256)
y_train = torch.tensor(train_df['author_id'].values, dtype=torch.long)
print(X_train)
print(y_train)
# Tokenize and encode the test data
X_test = tokenizer(list(test_df['text']), padding=True, truncation=True, return_tensors='pt', max_length=256)

# Train a Dutch BERT-based model
model = BertForSequenceClassification.from_pretrained('GroNLP/bert-base-dutch-cased', num_labels=len(label_encoder.classes_))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

true_labels = list(test_df['author'])

for epoch in range(epochs):
    outputs = model(**X_train, labels=y_train.unsqueeze(1))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Make predictions on the test data
with torch.no_grad():
    model.eval()
    logits = model(**X_test).logits

# Get predicted labels
predicted_labels = torch.argmax(logits, dim=1).numpy()

# Decode predicted labels
predicted_authors = label_encoder.inverse_transform(predicted_labels)

# Display the results
print(list(predicted_authors))
print(true_labels)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_authors, average='macro')
print('F1 Score:' + str(f1))
