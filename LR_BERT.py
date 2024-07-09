import lir
import argparse
import json
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import itertools
from helper_functions.BERT_helper import (BertMeanPoolingClassifier, validate_bert, finetune_bert,
                                          CustomDataset)
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from helper_functions.text_transformer import *
from helper_functions.split import split
from helper_functions.df_loader import load_df


def bert_model_scores(train, validate, train_labels, test_labels, config):
    # Assign device and check if it is GPU or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the model for fine-tuning
    # bert_model = RobertaModel.from_pretrained('BERTmodels/robbert-2023-dutch-base')
    bert_model = BertModel.from_pretrained('BERTmodels/bert-base-dutch-cased')
    model = BertMeanPoolingClassifier(bert_model, N_classes=2, dropout=config['BERT']['dropout'])
    model.to(device)

    train_labels = torch.tensor(list(train_labels), dtype=torch.long).to(device)

    # Set up DataLoader for training
    dataset = CustomDataset(train, train_labels)
    batch_size = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Fine-tuning and validation loop
    epochs = config['BERT']['epochs']
    model = finetune_bert(model, train_dataloader, epochs, config)
    preds, f1, scores = validate_bert(model, validate, test_labels)
    return scores

def LR_BERT(args,config):
    with open('config.json') as f:
        config = json.load(f)

    # Random Seed at file level
    random_seed = 33
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Assign device and check if it is GPU or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataframe
    full_df, config = load_df(args, config)


    # Encode author labels
    label_encoder = LabelEncoder()
    full_df['author'] = label_encoder.fit_transform(full_df['author'])

    n_authors = config['variables']['nAuthors']
    cllr_avg = np.zeros(4)

    # Limit the authors to nAuthors
    authors = list(set(full_df.author))
    reduced_df = full_df.loc[full_df['author'].isin(authors[:n_authors])]
    additional_df = full_df.loc[full_df['author'].isin(authors[n_authors:2 * n_authors])]
    constant_train_df = train_df.copy()
    df = reduced_df.copy()
    a = df['conversation'].unique()
    combinations = []
    for comb in itertools.combinations(a, 7):
        rest = list(set(a) - set(comb))
        combinations.append([list(comb), list(rest)])

    combinations = [([1, 2, 4, 5, 6, 7, 8], [3])]
    validation_lr = np.zeros(len(combinations) * n_authors ** 2)
    print(validation_lr[:n_authors])
    additional_lr = np.zeros(len(combinations) * n_authors ** 2)
    validation_truth = np.zeros(len(combinations) * n_authors ** 2)
    for i, comb in enumerate(combinations):
        print(i)
        del constant_train_df
        del train_df
        del test_df
        df = reduced_df.copy()

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.125, stratify=df[['author']])
        else:
            # For now only works without confusion
            train_df, test_df = split(df, 0.125, comb, confusion=False)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        test_df = pd.concat([test_df, additional_df[additional_df['conversation'] == comb[1][0]]])

        constant_train_df = train_df.copy()
        conversations = list(set(train_df['conversation']))
        lr_split = []
        true_split = []
        print(f"Test conversation: {comb[1]}")
        for suspect in range(0, len(set(train_df['author']))):
            train_df = constant_train_df.copy()
            train_df['h1'] = [int(1) if author == suspect else int(0) for author in train_df['author']]

            suspect_train_df = train_df.loc[train_df['author'] == suspect]
            for n in range(int(len(train_df['h1']) / len(suspect_train_df['h1'])) - 2):
                train_df = pd.concat([train_df, suspect_train_df])

            test_df['h1'] = [1 if author == suspect else 0 for author in test_df['author']]
            encoded_known_authors = test_df['h1']
            train_df = train_df.reset_index(drop=True)

            # Set tokenizer and tokenize training and test texts
            # tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
            tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')  #
            train_encodings = np.array(transform_list_of_texts(train_df['text'], tokenizer, 510, 256, 256, \
                                                               device=device))
            val_encodings = np.array(transform_list_of_texts(test_df['text'], tokenizer, 510, 256, 256, \
                                                             device=device))
            print('test')
            print(len(val_encodings))
            calibration_scores = np.zeros(len(conversations) * n_authors)
            calibration_truth = np.zeros(len(conversations) * n_authors)
            for j, c in enumerate(conversations):
                # c is conversation in calibration set, all others go in training set
                train = train_df.index[train_df['conversation'] != c].tolist()
                calibrate = train_df.index[train_df['conversation'] == c].tolist()

                calibration_encodings = train_encodings[calibrate][:n_authors]

                scores = bert_model_scores(train_encodings[train], calibration_encodings, train_df['h1'][train],
                                           config)
                print(scores)
                calibration_scores[j * n_authors:(j + 1) * n_authors] = scores
                calibration_truth[j * n_authors:(j + 1) * n_authors] = np.array(train_df['h1'][calibrate][:n_authors])

            validation_scores = bert_model_scores(train_encodings, val_encodings, train_df['h1'],
                                                  config)
            print(validation_scores)
            calibrator = lir.KDECalibrator(bandwidth='silverman')  # [0.01,0.1]
            calibrator.fit(calibration_scores, calibration_truth == 1)
            bounded_calibrator = lir.ELUBbounder(calibrator)
            bounded_calibrator.fit(calibration_scores, calibration_truth == 1)
            lrs_validation = bounded_calibrator.transform(validation_scores)
            k = i * n_authors ** 2 + suspect * n_authors
            validation_lr[k:k + n_authors] = lrs_validation[:n_authors]
            additional_lr[k:k + n_authors] = lrs_validation[n_authors:2 * n_authors]
            validation_truth[k:k + n_authors] = np.array(test_df['h1'])[:n_authors]

            time.sleep(0.01)
            cllr = lir.metrics.cllr(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
            cllr_min = lir.metrics.cllr_min(validation_lr[k:k + n_authors], validation_truth[k:k + n_authors])
            cllr_cal = cllr - cllr_min
            cllr_avg = cllr_avg + np.array([cllr, cllr_min, cllr_cal, 1])
            print(f"Average Cllr: {cllr_avg[0] / cllr_avg[3]:.3f}, Cllr_min: {cllr_avg[1] / cllr_avg[3]:.3f}\
                    , Cllr_cal: {cllr_avg[2] / cllr_avg[3]:.3f}")
            ones_list = np.ones(len(calibration_truth))

            with lir.plotting.show() as ax:
                ax.calibrator_fit(calibrator, score_range=[0, 1], resolution=1000)
                ax.score_distribution(scores=calibration_scores[calibration_truth == 1],
                                      y=ones_list[calibration_truth == 1],
                                      bins=np.linspace(0, 1, 9), weighted=True)
                ax.score_distribution(scores=calibration_scores[calibration_truth == 0],
                                      y=ones_list[calibration_truth == 0] * 0,
                                      bins=np.linspace(0, 1, 9), weighted=True)
                ax.xlabel('SVM score')
                H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='$H_1$-true')
                H2_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='$H_2$-true')
                ax.legend()
                plt.show()

        with lir.plotting.show() as ax:
            ax.tippett(validation_lr[i * n_authors ** 2:(i + 1) * n_authors ** 2],
                       validation_truth[i * n_authors ** 2:(i + 1) * n_authors ** 2])
        plt.show()

    print(f"Nauthors: {n_authors}")

    h1_lrs = validation_lr[validation_truth == 1]
    h2_lrs = validation_lr[validation_truth == 0]
    cllr = lir.metrics.cllr(validation_lr, validation_truth)
    cllr_min = lir.metrics.cllr_min(validation_lr, validation_truth)
    cllr_cal = cllr - cllr_min
    print(f"Cllr: {cllr:.3f}, Cllr_min: {cllr_min:.3f}, Cllr_cal: {cllr_cal:.3f}")
    print(f"Average Cllr: {cllr_avg[0] / cllr_avg[3]:.3f}, Cllr_min: {cllr_avg[1] / cllr_avg[3]:.3f}\
            , Cllr_cal: {cllr_avg[2] / cllr_avg[3]:.3f}")

    freq1 = np.histogram(h1_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h1_lrs)
    freq2 = np.histogram(h2_lrs, bins=[-np.inf] + [1, 100] + [np.inf])[0] / len(h2_lrs)
    print(f"H1 samples with LR < 1: {freq1[0] * 100:.3f}%, H2 samples with LR > 1: {(freq2[1] + freq2[2]) * 100:.3f}%")
    print(f"H1 samples with LR < 100: {(freq1[0] + freq1[1]) * 100:.3f}%, H2 samples with LR > 100: {freq2[2] * 100:.3f}%")
    print(f"H1 sample with lowest LR: {np.min(h1_lrs):.3f}, H2 sample with highest LR: {np.max(h2_lrs):.3f}")
    print(f"H1 sample with highest LR: {np.max(h1_lrs):.3f}, H2 sample with lowest LR: {np.min(h2_lrs):.3f}")
    """
    for j in range(n_authors):
        avg = np.zeros(n_authors)
        for i in range(len(combinations)):
            avg += np.log10(additional_lr[i*n_authors**2+j*n_authors:i*n_authors**2+(j+1)*n_authors])
        avg /= len(combinations)
        x = np.linspace(51,100,50)
        plt.scatter(x,avg)
        plt.show()
    """
    with lir.plotting.show() as ax:
        ax.tippett(validation_lr, validation_truth)
        lr_1 = np.log10(additional_lr)
        xplot1 = np.linspace(np.min(lr_1), np.max(lr_1), 100)
        perc1 = (sum(i >= xplot1 for i in lr_1) / len(lr_1)) * 100
        ax.plot(xplot1, perc1, color='g', label='LRs given $\mathregular{H_2}$ outside training set')
        ax.legend()
    plt.show()

    with lir.plotting.show() as ax:
        ax.pav(validation_lr, validation_truth)
    plt.show()

    with lir.plotting.show() as ax:
        ax.ece(validation_lr, validation_truth)
    plt.show()

    plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 1],
                np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 1])
    plt.scatter(validation_scores[:n_authors][validation_truth[k:k + n_authors] == 0],
                np.log10(validation_lr[k:k + n_authors])[validation_truth[k:k + n_authors] == 0])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Choose the path to the input")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help="Choose the path to the output")
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    LR_BERT(args, config)

if __name__ == '__main__':
    main()