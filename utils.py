"""
Helper functions
"""


def read_in_sentences(path_to_file):
    """
    :param path_to_file: the full path to the file containing sentences: one sentence per line
    :return: a list of strings
    """
    sentences = []
    with open(path_to_file, "r") as in_file:
        for line in in_file:
            line = line.split()  # split by white space and remove any newline characters
            # put the words back together by whitespace and append to the list and lower case all characters
            sentences.append(" ".join(line).lower())
            # TODO: something to consider: removing punctuation and numbers
    return sentences






