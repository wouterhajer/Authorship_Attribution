from sklearn.preprocessing import LabelEncoder
from helper_functions.text_transformer import transform_list_of_texts
import json
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
from helper_functions.BERT_helper import (BertMeanPoolingClassifier, CustomDataset, BertAverageClassifier, BertTruncatedClassifier,
                                          finetune_bert, validate_bert)
import argparse
from helper_functions.df_loader import load_df
from helper_functions.split import split
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
import numpy as np
import csv
import os
import gc


def combinations(conv, n):
    combinations = []
    for comb in itertools.combinations(conv, n):
        rest = list(set(conv) - set(comb))
        combinations.append([list(rest), list(comb)])
    return combinations


def BERT_crossvall(args, config):
    config['baseline'] = 0
    config['masking']['masking'] = 0
    # Assign device and check if it is GPU or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    l = 10

    # Load dataframe
    full_df, config = load_df(args, config)

    n_authors = config['variables']['nAuthors']

    # Limit the authors to nAuthors
    authors = list(set(full_df['author']))
    df = full_df.loc[full_df['author'].isin(authors[:n_authors])]
    print(len(df))

    # Encode author labels
    label_encoder = LabelEncoder()
    df['author_id'] = label_encoder.fit_transform(df['author'])
    pd.set_option('display.max_columns', None)

    # Find all conversation numbers and make all combinations of 7 in train set, 1 in test set
    combs = combinations(df['conversation'].unique(), 1)

    types = ['average']
    models = ['RobBERT']

    for type in types:
        config['BERT']['type'] = type
        for model in models:
            config['BERT']['model'] = model
            f1_score_total = np.zeros(l)

            # Set tokenizer and tokenize training and test texts
            if config['BERT']['model'] == 'RobBERT':
                tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
            elif config['BERT']['model'] == 'BERTje':
                tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
            # For every combination of conversations in train/test set, calculate the scores for true author,
            # conversation partner and other speakers
            for i, comb in enumerate(combs):
                print(i)
                print(comb)
                # Use random or deterministic split
                if bool(config['randomConversations']):
                    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
                else:
                    train_df, test_df = split(df, 0.25, comb, confusion=bool(config['confusion']))

                train_encodings, train_encodings_simple = transform_list_of_texts(train_df['text'], tokenizer, 510,
                                                                              256, 256, device=device)

                val_encodings, val_encodings_simple = transform_list_of_texts(test_df['text'], tokenizer, 510,
                                                                              256, 256, device=device)
                encoded_known_authors = label_encoder.transform(test_df['author'])
                train_labels = list(train_df['author_id'])
                train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
                N_classes = len(list(set(encoded_known_authors)))

                # Define the model for fine-tuning
                if config['BERT']['model'] == 'RobBERT' :
                    bert_model = RobertaModel.from_pretrained('BERTmodels/robbert-2023-dutch-base')
                elif config['BERT']['model'] == 'BERTje' :
                    bert_model = BertModel.from_pretrained('BERTmodels/bert-base-dutch-cased')


                if config['BERT']['type'] == 'meanpooling':
                    model = BertMeanPoolingClassifier(bert_model, device, N_classes=N_classes,
                                                      dropout=config['BERT']['dropout'])
                elif config['BERT']['type'] == 'truncated':
                    model = BertTruncatedClassifier(bert_model, device, N_classes=N_classes, dropout=config['BERT']['dropout'])
                    train_encodings, val_encodings = train_encodings_simple, val_encodings_simple
                elif config['BERT']['type'] == 'average':
                    model = BertAverageClassifier(bert_model, device, N_classes=N_classes, dropout=config['BERT']['dropout'])
                else:
                    print("No BERT model specified")

                model.to(device)

                # Set up DataLoader for training
                dataset = CustomDataset(train_encodings, train_labels)
                batch_size = 1
                train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Fine-tuning and validation loop
                epochs = 5
                for j in range(l):
                    print('hello')
                    model = finetune_bert(model, train_dataloader, epochs, config)

                    print('validation set')
                    preds, f1, scores = validate_bert(model, val_encodings, encoded_known_authors)

                    f1_score_total[j] += f1

                    """
                    author_number = [author for author in test_df['author']]
                    conf = confusion_matrix(test_df['author'], avg_preds, normalize='true')
                    cmd = ConfusionMatrixDisplay(conf, display_labels=sorted(set(author_number)))
                    cmd.plot()
                    plt.show()
                    """
                print(f1_score_total[j] / (i + 1) )

                del bert_model
                del model
                del train_encodings, train_encodings_simple, val_encodings, val_encodings_simple
                del train_dataloader
                del train_df, test_df
                # Then clean the cache
                torch.cuda.empty_cache()
                # then collect the garbage
                gc.collect()

            output_file = args.output_path + os.sep + 'BERT_' + args.corpus_name + ".csv"
            with open(output_file, 'a', newline='') as file:
                writer = csv.writer(file)
                aa_score_partner = f1_score_total/(i+1)
                writer.writerow([round(aa_score_partner[0], 3), round(aa_score_partner[1], 3), \
                                 round(aa_score_partner[2], 3), round(aa_score_partner[3], 3), \
                                 round(aa_score_partner[4], 3), round(aa_score_partner[5], 3), \
                                 round(aa_score_partner[6], 3), \
                                 round(aa_score_partner[7], 3), round(aa_score_partner[8], 3), \
                                 round(aa_score_partner[9], 3), config['confusion'], \
                                 args.corpus_name, config['BERT']['epochs'], config['BERT']['type'],
                                config['BERT']['model']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to the output")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)

    # Download new bertmodel from huggingface and save locally
    """
    model = RobertaModel.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
    model.save_pretrained('BERTmodels/robbert-2023-dutch-base')
    """
    BERT_crossvall(args, config)


if __name__ == '__main__':
    main()