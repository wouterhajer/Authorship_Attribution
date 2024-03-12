import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from char_dist import *
from sklearn.metrics import f1_score
from df_creator import *
from split import *
import itertools
from ngram import ngram
from vocabulary import *
import csv

if __name__ == '__main__':
    start = time.time()

    # Load config file
    with open('config.json') as f:
        config = json.load(f)

    # Random Seed at file level
    random_seed = 37
    np.random.seed(random_seed)
    random.seed(random_seed)

    full_df, train_df, test_df, background_vocab = create_df('txt', config)

    # Encode author labels
    label_encoder = LabelEncoder()
    train_df['author'] = label_encoder.fit_transform(train_df['author'])
    test_df['author'] = label_encoder.transform(test_df['author'])
    df = pd.concat([train_df,test_df])
    """
    vocab_word = []
    with open('Frequenties.csv', newline='') as csvfile:
        vocab_words = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in vocab_words:

            vocab_word.append(row[0].lower())
            if i > 50000:
                break
            i += 1
    print(vocab_word[900:1000])
    background_vocab = vocab_word[1:]
    """
    print(background_vocab[:100])
    print(background_vocab[400:500])
    # Find all conversation numbers and make all combinations of 6 in train set, 2 in test set
    a = df['conversation'].unique()
    combinations = []
    for comb in itertools.combinations(a, 6):
        rest = list(set(a) - set(comb))
        combinations.append([list(comb), list(rest)])

    score = 0
    score_partner = 0
    score_rest = 0
    score_sure = 0
    f1_total = 0

    # For every combination of conversations in train/test set, calculate the scores for true author,
    # conversation partner and other speakers
    for i, comb in enumerate(combinations):
        print(i)

        # Use random or deterministic split
        if bool(config['randomConversations']):
            train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[['author']])
        else:
            train_df, test_df = split(df, 0.25, comb, confusion=bool(config['confusion']))

        # Train SVMs, calculate predictions
        avg_preds, preds_char, preds_word, test_authors, sure = ngram(train_df, test_df, config, background_vocab)

        # Inverse label encodings
        avg_preds = label_encoder.inverse_transform(avg_preds)
        preds_char = label_encoder.inverse_transform(preds_char)
        preds_word = label_encoder.inverse_transform(preds_word)
        test_authors = label_encoder.inverse_transform(test_authors)

        # When using baseline, only word prediction counts
        if bool(config['baseline']):
            avg_preds = preds_word

        # Calculate the scores
        for j in range(len(test_df['author'])):
            if test_authors[j] == avg_preds[j]:
                score += 1
                if sure[j]:
                    score_sure += 1
            elif test_authors[j] == avg_preds[j] + 1 and list(test_df['author'])[j] % 2 == 1:
                score_partner += 1
            elif test_authors[j] == avg_preds[j] - 1 and list(test_df['author'])[j] % 2 == 0:
                score_partner += 1
            else:
                score_rest += 1

        n_prob = len(sure)
        n_auth = len(set(test_authors))
        f1 = f1_score(test_authors, avg_preds, average='macro')

        f1_total += f1
        # Print the scores at this iteration
        print("Score = {:.4f}, random chance = {:.4f} ".format(score / n_prob / (i + 1), 1 / n_auth))
        print("Score partner = {:.4f}, random chance = {:.4f} ".format(score_partner / n_prob / (i + 1), 1 / n_auth))
        print("Score rest = {:.4f}, random chance = {:.4f} ".format(score_rest / n_prob / (i + 1), 1 - 2 / n_auth))
        print("F1-score = {:.4f}".format(f1))
        print("Average F1-score = {:.4f}".format(f1_total / (i + 1)))

    print('Included authors: ' + str(len(set(test_authors))))

    """
    # Calculate F1 score
    f1 = f1_score(test_authors, avg_preds, average='macro')
    print('F1 Score average:' + str(f1))

    f1 = f1_score(test_authors, preds_char, average='macro')
    print('F1 Score char:' + str(f1))

    # f1 = f1_score(test_authors, preds_char_dist, average='macro')
    # print('F1 Score char dist:' + str(f1))

    f1 = f1_score(test_authors, preds_word, average='macro')
    print('F1 Score word:' + str(f1) + '\n')
    
    
    
    # Print whether or not the model was sure
    print('Percentage sure: ' + str(sure.count(True) / len(sure)))
    print('Score when sure: ' + str(score_sure / sure.count(True)))
    indeces = [i for i, x in enumerate(sure) if x]
    f1 = f1_score([test_authors[x] for x in indeces], [avg_preds[x] for x in indeces], average='macro')
    print('Macro F1 when sure:' + str(f1))
    indeces = [i for i, x in enumerate(sure) if not x]
    f1 = f1_score([test_authors[x] for x in indeces], [avg_preds[x] for x in indeces], average='macro')
    print('Macro F1 when unsure:' + str(f1) + '\n')
    """
    # Print duration
    print('Total time: ' + str(time.time() - start) + ' seconds')
