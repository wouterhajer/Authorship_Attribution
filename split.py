import random
import pandas as pd

def split(df2, p, conversations=None, confusion=False):
    """
    Function to split dataframe into train and test based on conversations number in Frida
    :param df: Original dataframe
    :param p: Percentage of conversations in test set
    :param conversations: Specific, non random split (tuple)
    :return: train and test dataframe
    """
    df = df2.copy()
    # If no specific conversations are given, decide randomly
    if conversations == None:
        a = df['conversation'].unique()
        random.shuffle(a)
        train_conv = a[:int(len(a) * (1 - p))]
        test_conv = a[int(len(a) * (1 - p)):]
    else:
        a = conversations
        train_conv = a[0]
        test_conv = a[1]
    print(train_conv,test_conv)
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
        train_conv, test_conv = abc_nl1_convert(test_conv) #frida_convert(test_conv)
        train_df_even = even_df[even_df['conversation'].isin(train_conv)]
        test_df_even = even_df[even_df['conversation'].isin(test_conv)]
        train_df = pd.concat([train_df_odd, train_df_even])
        test_df = pd.concat([test_df_odd, test_df_even])
    return train_df, test_df

def frida_convert(conv):
    """
    Determines the conversation in the test set for even authors such that they are in the training set for the odd
    authors. Also assigns the corresponding training set for the even authors.
    :param conv: test conversations for the odd authors
    :return: training and test conversations for the even authors
    """
    print(conv)
    test_conv = []
    if conv[0] % 8 != conv[1] - 1:
        test_conv.append((conv[0]) % 8 + 1)
        if conv[1] % 8 != conv[0] - 1:
            test_conv.append((conv[1]) % 8 + 1)
        else:
            test_conv.append((conv[1] + 2) % 8 + 1)
    else:
        test_conv.append((conv[0] + 1) % 8 + 1)
        test_conv.append((conv[1] + 1) % 8 + 1)
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    train_conv = list(set(a) - set(test_conv))
    return train_conv, test_conv

def abc_nl1_convert(conv):
    """
    Determines the conversation in the test set for even authors such that they are in the training set for the odd
    authors. Also assigns the corresponding training set for the even authors.
    :param conv: test conversations for the odd authors
    :return: training and test conversations for the even authors
    """
    n = 6
    print('hello')
    print(conv)
    test_conv = [(conv[0])%n+1]

    a = [1, 2, 3, 4, 5, 6] #[1, 2, 3, 4, 5, 6, 7, 8]
    train_conv = list(set(a) - set(test_conv))
    return train_conv, test_conv

