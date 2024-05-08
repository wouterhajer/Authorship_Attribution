import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import gc
import numpy as np

class BertSimpleClassifier(nn.Module):
    """
    Bert model with added mean pooling layer and dense layer to handle longer texts.
    """
    def __init__(self, bert_model, N_classes, dropout=0.5):
        super(BertSimpleClassifier, self).__init__()
        self.bert_model = bert_model #BertModel.from_pretrained('BERTmodels/bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(in_features=768, out_features=N_classes)

    def forward(self, encodings):
        # Obtain BERT hidden states
        inputs = self.bert_model(**encodings)['last_hidden_state']
        output = inputs
        # Apply dropout on the dense layer and pass through dense layer
        pooled_output = self.dropout(output)
        logits = self.dense(pooled_output[0,0])
        return logits

    def inference(self, encodings):
        # Obtain BERT hidden states
        inputs = self.bert_model(**encodings)['last_hidden_state']

        # Pass through dense layer
        logits = self.dense(inputs[0,0])
        return logits


class CustomDataset(Dataset):
    """
    Dataset class to handle longer texts in a form that pytorch can handle during training.
    """
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

def finetune_bert_simple(model, train_dataloader, epochs, config):
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BERT']['learningRate'])
    criterion = torch.nn.CrossEntropyLoss()
    # Currently using Number of authors as batch size
    N_classes = config['variables']['nAuthors']
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i = 0
        for batch in train_dataloader:
            encoding, labels = batch['encodings'], batch['labels'][0]
            encoding = {'input_ids': encoding['input_ids'][0],
                        'token_type_ids': encoding['token_type_ids'][0],
                        'attention_mask': encoding['attention_mask'][0]}
            outputs = model(encoding)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            if i % N_classes == N_classes - 1:
                optimizer.step()
                optimizer.zero_grad()
            i += 1

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")
        # delete locals
        del encoding
        del outputs
        del loss
        # Then clean the cache
        torch.cuda.empty_cache()
        # then collect the garbage
        gc.collect()
    del train_dataloader
    del optimizer
    del criterion
    # Then clean the cache
    torch.cuda.empty_cache()
    # then collect the garbage
    gc.collect()
    return model

def validate_bert_simple(model, val_encodings, encoded_known_authors):
    """
    :param model: The finetuned BertMeanPoolingClassifier to be evaluated
    :param val_encodings: Encodings of validation texts, split in overlapping chunks of 512 tokens
    :param encoded_known_authors: The real encoded authors of each validation text.
    :return: predictions for each text in the validation set
    Validation loop for meanpooling BERT calculating a F1-score and returning predictions for each text.
    """
    preds = []
    model.eval()
    with torch.no_grad():
        for i,encoding in enumerate(val_encodings):
            # Tokenize and encode the validation data
            output = model.inference(encoding)
            val_predictions = torch.argmax(output, dim=0)
            preds.append(val_predictions.detach().cpu().numpy())

    f1 = f1_score(encoded_known_authors, preds, average='macro')
    print('F1 Score average:' + str(f1))
    return preds, f1
