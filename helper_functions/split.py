import random
import pandas as pd
import numpy as np
import itertools


def combinations(convs, crossval):
    n_conv = len(convs)
    if crossval:
        combinations = []
        for comb in itertools.combinations(convs, n_conv - 1):
            rest = list(set(convs) - set(comb))
            combinations.append([list(comb), list(rest)])
    else:
        convs_list = list(convs)
        combinations = [(convs_list[:-1], [convs_list[-1]])]
    return combinations

def split(df, p=0.125, conversations=None, confusion=False):
    """
    Function to split dataframe into train and test based on conversations number in Frida
    :param df: Original dataframe
    :param p: Percentage of conversations in test set
    :param conversations: Specific, non-random split (tuple)
    :return: train and test dataframe
    """

    # If no specific conversations are given, decide randomly
    if conversations == None:
        a = df['conversation'].unique()
        print(a)
        random.shuffle(a)
        train_conv = a[:int(len(a) * (1 - p))]
        test_conv = a[int(len(a) * (1 - p)):]
    else:
        a = conversations
        train_conv = a[0]
        test_conv = a[1]

    # Split the dataframe
    train_df = df[df['conversation'].isin(train_conv)]
    test_df = df[df['conversation'].isin(test_conv)]

    # If confusion is true, ensure conversation of odd speakers in test set are in training set of even speakers,
    # and vice versa
    if confusion:
        odd_df = df.loc[df['author'] % 2 == 1]
        even_df = df.loc[df['author'] % 2 == 0]
        train_df_odd = odd_df[odd_df['conversation'].isin(train_conv)]
        test_df_odd = odd_df[odd_df['conversation'].isin(test_conv)]
        len_conv = len(train_conv)+len(test_conv)

        train_conv, test_conv = convert_confusion(test_conv, len_conv)

        train_df_even = even_df[even_df['conversation'].isin(train_conv)]
        test_df_even = even_df[even_df['conversation'].isin(test_conv)]
        train_df = pd.concat([train_df_odd, train_df_even])
        test_df = pd.concat([test_df_odd, test_df_even])
    return train_df, test_df

def convert_confusion(conv,len_conv):
    """
    Determines the conversation in the test set for even authors such that they are in the training set for the odd
    authors. Also assigns the corresponding training set for the even authors.
    :param conv: test conversations for the odd authors
    :return: training and test conversations for the even authors
    """
    test_conv = np.zeros(len(conv))
    for i in range(len(conv)):
        # create list  of conversations already in test set or in previous test set
        conv_copy = np.concatenate((conv.copy(), test_conv))
        # find the next conversation not in use and add to new test conversations
        j = 0
        while j < 100:
            if (conv[i]+j) % len_conv + 1 not in conv_copy:
                test_conv[i] = (conv[i]+j) % len_conv + 1
                break
            j += 1
        if j == 100:
            print('No new conversation found')
            test_conv = test_conv[:i]
            continue

    a = [i + 1 for i in range(len_conv)]
    train_conv = list(set(a) - set(test_conv))
    return train_conv, test_conv

