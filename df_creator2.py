import os
import glob
import codecs
import json
from split import *
from vocabulary import extend_vocabulary
import argparse
import re

def create_df(args):
    """
    :param args: Arguments containing corpus_path, corpus_name and output_path
    :return: Dataframe containing column 'text' for all texts and 'author' with corresponding labels for the author

    This function is outputs a dataframe with colums text, author and conversation for two of the studied corpora.
    The dataframe is saved to a CSV file at the output_path.
    """
    # Reads all text files located at the corpus_path
    files = glob.glob(args.corpus_path + os.sep + '*.txt')
    texts = []

    # Specifics for the FRIDA corpus
    if args.corpus_name == "Frida":
        # Hard code conversation numbers for speaker with an odd or even number such that conversation match
        odd = [1, 2, 3, 4, 5, 6, 7, 8]
        even = [3, 4, 1, 2, 7, 8, 5, 6]

        # keep track of the author of the last document
        author = 0

        for i, v in enumerate(sorted(files)):
            # Get author number from filename
            new_author = int(v[len(args.corpus_path)+3:len(args.corpus_path)+6])

            # Select the first recording of each text and ignore authors with subscript a
            if v[-5] == '1' and 310 > new_author > 0 and v[len(args.corpus_path)+6] != 'a':
                # Check if author corresponds to current author and count the conversation
                if author != new_author:
                    author = new_author
                    j = 0
                else:
                    j += 1

                # Assign the conversation number based on whether the author number is odd or even
                if author % 2 == 1:
                    conversation = odd[j]
                else:
                    conversation = even[j]

                f = codecs.open(v, 'r', encoding='utf-8')
                label = new_author
                text = f.read()

                # Make a single text from the lines of the transcript, disregard the timestamps
                text = text.split('\r\n')
                text2 = []
                for lines in text[1:-1]:
                    line = lines.split('\t')
                    text2.append(line[2])
                text3 = ' '.join(text2[:])

                # Some conversations are ended prematurely, ignore them
                if len(text3) < 200:
                    print(text3)
                    print(v)
                    continue

                # Append result as tuple to list
                texts.append((text3, label, conversation))
                f.close()

    # Specifics for the abc_nl1 corpus
    elif args.corpus_name == "abc_nl1":
        for i, v in enumerate(sorted(files)):
            f = codecs.open(v, 'r', encoding=None)

            text = f.read()
            text = text.split('  ')
            text2 = []
            for j, line in enumerate(text):
                if line != '':
                    text2.append(line)

            text3 = ' '.join(text2[:])
            text3 = re.sub(r'\n\n\n', ' ', text3)
            text3 = re.sub(r'\n\n', ' ', text3)
            text3 = re.sub(r'\n', ' ', text3)
            author = int(v[15])
            conv = int(v[18])
            texts.append((text3, author, conv))

    # Convert into dataframe
    df = pd.DataFrame(texts, columns=['text', 'author', 'conversation'])

    # Create specific vocabulary corresponding to the dataframe and save to txt file
    background_vocab = extend_vocabulary([1, 1], df['text'], model='word')
    vocab_file = args.output_path+os.sep+'vocab_'+args.corpus_name+".txt"
    with open(vocab_file, 'w', encoding="utf-8") as fp:
        fp.write('\n'.join(background_vocab))

    # Save data frame to csv file
    df.to_csv(args.output_path+os.sep+args.corpus_name+".csv", index=False)

    return df, background_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', help = "Choose the path to the corpus")
    parser.add_argument('corpus_name', help="Choose the name of the corpus")
    parser.add_argument('output_path', help = "Choose the path to output folder")
    args = parser.parse_args()
    df, vocab = create_df(args)

if __name__ == '__main__':
    main()
