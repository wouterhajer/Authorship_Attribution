def mask(df, vocab_word, config):
    """
    Masks texts in dataframe by replacing out of vocab words with q
    :param df: dataframe containing texts and labels
    :param vocab_word: list containing all words to be kept
    :param config: dictionary containing variable values
    :return: dataframe after masking
    """
    df = df.reset_index(drop=True)
    single_masking = bool(config['masking']['singleMasking'])
    for k, text in enumerate(df['text']):
        i = 0
        while i < len(text):
            if not text[i].isalpha():
                i += 1
            else:
                j = i + 1
                if not j == len(text):
                    while text[j] != ' ':
                        j += 1
                        if j == len(text):
                            break
                word = text[i:j]
                # Remove punctuation to fit word vocabulary
                string = ''.join([char for char in word if char.isalnum()])

                if string.lower() not in vocab_word:
                    if single_masking:
                        text = text[:i] + '#' + text[j:]
                        i += 1
                    else:
                        text = text[:i] + '#' * len(word) + text[j:]
                        i = j
                else:
                    i = j
        df.loc[k, 'text'] = text
    return df
