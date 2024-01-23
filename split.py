import random



def split(df, p, conversations = None):
    if conversations == None:
        a = df['conversation'].unique()
        random.shuffle(a)
        train_conv = a[:int(len(a) * (1 - p))]
        test_conv = a[int(len(a) * (1 - p)):]
    else:
        a = conversations
        train_conv = a[0]
        test_conv = a[1]

    train_df = df[df['conversation'].isin(train_conv)]
    test_df = df[df['conversation'].isin(test_conv)]

    return train_df, test_df
