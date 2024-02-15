def mask(df, vocab_word, config):
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
                    while text[j].isalpha():
                        j += 1
                        if j == len(text):
                            break
                word = text[i:j]
                if word.lower() not in vocab_word:
                    if single_masking:
                        text = text[:i] + 'q' + text[j:]
                        i += 1
                    else:
                        text = text[:i] + 'q' * len(word) + text[j:]
                        i = j
                else:
                    i = j
        df.loc[k, 'text'] = text
    #print(df)
    return df
