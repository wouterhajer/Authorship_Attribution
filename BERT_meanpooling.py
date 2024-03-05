import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

class BertMeanPoolingClassifier(nn.Module):
    def __init__(self, N_classes, dropout=0.5):
        super(BertMeanPoolingClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('BERTmodels/bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(in_features=768, out_features=N_classes)
    def forward(self, encodings):
        # Obtain BERT hidden states
        inputs = self.bert_model(**encodings)['last_hidden_state']

        # Mean pooling across the sequence dimension
        pooled_output = torch.mean(inputs, dim=0)

        # Apply dropout and pass through dense layer
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output[0])
        return logits

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'encodings': self.encodings[idx], 'labels': self.labels[idx]}
        return sample

def validate(model, val_encodings, encoded_known_authors):
    preds = []
    model.eval()
    with torch.no_grad():
        for i,label in enumerate(encoded_known_authors):
            # Tokenize and encode the validation data
            output = model(val_encodings[i])
            val_predictions = torch.argmax(output, dim=0)
            preds.append(val_predictions.detach().cpu().numpy())

    f1 = f1_score(encoded_known_authors, preds, average='macro')
    print('F1 Score average:' + str(f1))
    return preds
