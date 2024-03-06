import torch
from sklearn.metrics import f1_score
import gc

# still need to import the right functions from the colab file

def finetune_bert_simple(model, train_dataloader, epochs, config):
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BERT']['learningRate'])
    criterion = torch.nn.CrossEntropyLoss()
    # Currently using Number of authors as batch size
    N_classes = config['Pan2019']['nClasses']
    #epochs = config['BERT']['epochs']
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i = 0
        for batch in train_dataloader:
            encoding, labels = batch['encodings'], batch['labels'][0]

            encoding = {'input_ids': encoding['input_ids'][0],
                        'token_type_ids': encoding['token_type_ids'][0],
                        'attention_mask': encoding['attention_mask'][0]
                        }

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
        for i,label in enumerate(val_encodings):
            # Tokenize and encode the validation data
            output = model.inference(val_encodings[i])
            val_predictions = torch.argmax(output, dim=0)
            preds.append(val_predictions.detach().cpu().numpy())

    f1 = f1_score(encoded_known_authors, preds, average='macro')
    print('F1 Score average:' + str(f1))
    return preds
