import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from text_transformer import transform_list_of_texts
import json
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from BERT_meanpooling import BertMeanPoolingClassifier, finetune_bert_meanpooling, validate_bert_meanpooling, CustomDataset
from BERT_simple import finetune_bert_simple, validate_bert_simple, BertSimpleClassifier
import argparse
from df_loader import load_df
from split import split
from sklearn.model_selection import train_test_split
import pandas as pd

def BERT(args, config):
    # Assign device and check if it is GPU or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Download new bertmodel from huggingface and save locally
    """
    model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    model.save_pretrained('BERTmodels/bert-base-dutch-cased')
    """
    # Load dataframe
    full_df, config = load_df(args, config)

    n_authors = config['variables']['nAuthors']
    print(full_df)

    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author_id'] = label_encoder.fit_transform(full_df['author'])
    pd.set_option('display.max_columns', None)

    # Limit the authors to nAuthors
    authors = list(set(full_df.author_id))
    reduced_df = full_df.loc[full_df['author_id'].isin(authors[:n_authors])]

    if bool(config['randomConversations']):
        train_df, test_df = train_test_split(reduced_df, test_size=0.25, stratify=reduced_df[['author']])
    else:
        print('hello')
        train_df, test_df = split(reduced_df, 0.25, confusion=bool(config['confusion']))
    print(train_df)
    print(test_df)
    # Encode author labels
    label_encoder = LabelEncoder()
    train_df['author_id'] = label_encoder.fit_transform(train_df['author'])
    # Limit the authors to nAuthors
    authors = list(set(full_df.author))
    train_df = train_df.loc[train_df['author'].isin(authors[:n_authors])]
    test_df = test_df.loc[test_df['author'].isin(authors[:n_authors])]

    # Set tokenizer and tokenize training and test texts
    # tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
    tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')  #
    train_encodings, train_encodings_simple = transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256, \
                                        device=device)
    val_encodings, val_encodings_simple = transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256, \
                                            device=device)

    encoded_known_authors = label_encoder.transform(test_df['author'])
    train_labels = torch.tensor(train_df['author_id'], dtype=torch.long).to(device)
    N_classes = len(list(set(encoded_known_authors)))

    # Define the model for fine-tuning
    #bert_model = RobertaModel.from_pretrained('BERTmodels/robbert-2023-dutch-base')
    bert_model = BertModel.from_pretrained('BERTmodels/bert-base-dutch-cased')
    #model = BertMeanPoolingClassifier(bert_model, N_classes=N_classes, dropout=config['BERT']['dropout'])
    model = BertSimpleClassifier(bert_model, N_classes=N_classes, dropout=config['BERT']['dropout'])
    model.to(device)

    # Set up DataLoader for training
    if config['BERT']['type'] == 'truncated':
        train_encodings, val_encodings = train_encodings_simple, val_encodings_simple

    dataset = CustomDataset(train_encodings, train_labels)
    batch_size = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fine-tuning and validation loop
    epochs = 1
    #preds, scores = validate_bert_simple(model, val_encodings, encoded_known_authors)
    for j in range(10):

        #model = finetune_bert_meanpooling(model, train_dataloader, epochs, config)
        model = finetune_bert_simple(model, train_dataloader, epochs, config)

        print('validation set')
        #preds, scores = validate_bert_meanpooling(model, val_encodings, encoded_known_authors)
        preds, scores = validate_bert_simple(model, val_encodings, encoded_known_authors)
        avg_preds = label_encoder.inverse_transform(preds)
        author_number = [author for author in test_df['author']]
        conf = confusion_matrix(test_df['author'], avg_preds, normalize='true')
        #cmd = ConfusionMatrixDisplay(conf, display_labels=sorted(set(author_number)))
        #cmd.plot()
        #plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to the output")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    BERT(args, config)

if __name__ == '__main__':
    main()
