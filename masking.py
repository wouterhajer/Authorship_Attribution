def mask(df, vocab_word, config):
    single_masking = bool(config['masking']['singleMasking'])
    for k, text in enumerate(df['text']):
        print(len(text))
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
                if word not in vocab_word:
                    if single_masking:
                        text = text[:i] + '*' + text[j:]
                    else:
                        text = text[:i] + '*' * len(word) + text[j:]
                i = j
        df['text'].iloc[k] = text
    return df
