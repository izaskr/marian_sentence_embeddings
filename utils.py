"""
Helper functions
"""
import torch
import torch.nn as nn
import numpy as np
from wordfreq import word_frequency

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


def surprisal_per_word_from_scratch(sentence, correcting, device, model_id, tokenizer, model):

    logsoftmax = nn.LogSoftmax()

    all_surprisals = []
    all_tokens = []
    corrected_tokens = []
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)[0]

    target_ids = input_ids.clone()

    # get model outputs
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    logits = outputs.logits
    for i in range(len(input_ids)):
        current_token = tokenizer.decode(input_ids[i].item())
        all_tokens.append(current_token)
        if i == 0:
            # print("Current token ", current_token)
            unigram_frequency = word_frequency(current_token.strip(), "de")
            if unigram_frequency == 0:
                pseudo_surprisal = -1 * np.log(0.000000000000000000000000000000000000000000000000000000000001)
                print("String-inital subword has surprisal", pseudo_surprisal)
            if unigram_frequency != 0:
                pseudo_surprisal = -1 * np.log(unigram_frequency)
            all_surprisals.append(pseudo_surprisal)
            continue

        current_logits = logits[i - 1]
        current_logprob = logsoftmax(current_logits)
        surprisal_actual_token = -1 * current_logprob[input_ids[i]]
        all_surprisals.append(surprisal_actual_token.item())

    if correcting:
        corrected_surprisals = []
        corrected_tokens = []
        # join the surprisals of subwords that belong to the same word
        # if a token does not start with a white-space, it belongs together to the previous token
        # if the token is a punctuation mark, treat it as a full word
        for j, t in enumerate(all_tokens):
            if t == "." or t == ",":
                corrected_surprisals.append(all_surprisals[j])
                corrected_tokens.append(t)
                continue
            if j == 0:
                corrected_surprisals.append(all_surprisals[j])
                corrected_tokens.append(t)
                continue
            if not t.startswith(" "):
                before_joining_surprisal = all_surprisals[j]
                corrected_s = corrected_surprisals[-1] + before_joining_surprisal
                corrected_surprisals[-1] = corrected_s
                before_joining_token = corrected_tokens[-1]
                corrected_tokens[-1] = before_joining_token + t
            if t.startswith(" "):
                corrected_surprisals.append(all_surprisals[j])
                corrected_tokens.append(t)

        return corrected_surprisals, all_tokens, corrected_tokens

    return all_surprisals, all_tokens



