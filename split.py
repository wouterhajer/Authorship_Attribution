import random

def split(df, p, conversations = None):
    """
    Function to split dataframe into train and test based on conversations number in Frida
    :param df: Original dataframe
    :param p: Percentage of conversations in test set
    :param conversations: Specific, non random split (tuple)
    :return: train and test dataframe
    """
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

    # Split the dataframe
    train_df = df[df['conversation'].isin(train_conv)]
    test_df = df[df['conversation'].isin(test_conv)]

    return train_df, test_df
