import argparse
import json
import pandas as pd
import os
from text_transformer import transform_list_of_texts
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer


def add_embeddings(args,config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_file = args.input_path + os.sep + args.corpus_name + ".csv"
    full_df = pd.read_csv(df_file)
    """
    # Set tokenizer and tokenize training and test texts
    # tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
    #tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    tokenizer = AutoTokenizer.from_pretrained('AnnaWegmann/Style-Embedding')
    encodings, encodings_simple = transform_list_of_texts(full_df['text'], tokenizer, 510,
                                                                256, 256, device=device)
    
    # Define the model for fine-tuning
    #bert_model = RobertaModel.from_pretrained('BERTmodels/robbert-2023-dutch-base')
    #bert_model = BertModel.from_pretrained('BERTmodels/bert-base-dutch-cased')
    bert_model = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding')
    bert_model.to(device)
    
    state_simple = []
    state = []
    for encoding in encodings:
        inputs = bert_model(**encoding)['last_hidden_state'][:, 0].detach().cpu().tolist()
        state_simple.append(str(inputs[0]))
        average = list(np.mean(inputs, axis=0))
        state.append(str(average))
    print(inputs)
    
    print(len(inputs))
    """
    model = SentenceTransformer('BERTmodels/Style_Embedding')
    embeddings = model.encode(list(full_df['text']))
    print(embeddings)
    full_df['embedding'] = embeddings
    full_df['embedding_simple'] = embeddings
    full_df.to_csv(args.output_path + os.sep + args.corpus_name + ".csv", index=False)
    return full_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help = "Choose the path to the dataframe")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help = "Choose the path to output folder")
    args = parser.parse_args()

    # Download new bertmodel from huggingface and save locally

    model = SentenceTransformer('AnnaWegmann/Style-Embedding')
    model.save_pretrained('BERTmodels/Style-Embedding')

    with open('config.json') as f:
        config = json.load(f)
    df = add_embeddings(args,config)
    print(df[:50])


if __name__ == '__main__':
    main()