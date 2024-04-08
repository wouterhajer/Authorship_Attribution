import re

"""
Modified from Boeninghoff et all
"""

def regex(string: str, model: str):
    """
    Function that computes regular expressions.
    """
    string = re.sub("[0-9]", "0", string)  # each digit will be represented as a 0
    string = re.sub(r'( \n| \t)+', '', string)
    # text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)

    if model == 'word':
        # if model is a word n-gram model, remove all punctuation
        string = ''.join([char for char in string if char.isalnum()])

    if model == 'char-dist':
        string = re.sub("[a-zA-Z]+", "*", string)
        # string = ''.join(['*' if char.isalpha() else char for char in string])

    return string


def frequency(tokens: list):
    """
    Count tokens in text (keys are tokens, values are their corresponding frequencies).
    """
    freq = dict()
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1
    return freq


def represent_text(text, n: int, model: str):
    """
    Extracts all character or word 'n'-grams from a given 'text'.
    Any digit is represented through a 0.
    Each hyperlink is replaced by an @ sign.
    The latter steps are computed through regular expressions.
    """
    if model == 'char-std' or model == 'char-dist':

        text = regex(text, model)
        tokens = [text[i:i + n] for i in range(len(text) - n + 1)]

        if model == 'char-std' and n == 2:
            # create list of unigrams that only consists of punctuation marks
            # and extend tokens by that list
            punct_unigrams = [token for token in text if not token.isalnum()]
            tokens.extend(punct_unigrams)

    elif model == 'word':
        # Either use re.split(' |[*]|-', text) or re.split(' ', text) depending on the interpretation of uh, v, s and u
        text = [regex(word, model) for word in re.split(' ', text) if regex(word, model)]
        tokens = [' '.join(text[i:i + n]) for i in range(len(text) - n + 1)]

    freq = frequency(tokens)

    return freq


def extract_vocabulary(texts: list, n: int, ft: int, model: str):
    """
    Extracts all character 'n'-grams occurring at least 'ft' times in a set of 'texts'.
    """
    occurrences = {}

    for text in texts:

        text_occurrences = represent_text(text, n, model)

        for ngram in text_occurrences.keys():

            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]

    vocabulary = sorted(occurrences, key=occurrences.get, reverse=True)
    return vocabulary

def extend_vocabulary(n_range: tuple, texts: list, model: str):
    n_start, n_end = n_range
    vocab = []
    for n in range(n_start, n_end + 1):
        n_vocab = extract_vocabulary(texts, n, (n_end - n) + 1, model)
        vocab.extend(n_vocab)
    return vocab