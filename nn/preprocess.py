# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import re
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # we can use the positive classes as they are (all of them) and just sample the same number of negative classes from the large pool.
    all_pos_seqs=[]
    all_pos_labels=[]
    all_neg_seqs=[]

    # isolate positive and negativeclasses
    for seq, label in zip(seqs, labels):
        if (label==True):
            all_pos_seqs.append(seq)
            all_pos_labels.append(label)
        else:
            all_neg_seqs.append(seq)

    # randomly sample negative classes in the same amount as there are positive classes
    neg_seqs_sample=random.sample(all_neg_seqs, len(all_pos_seqs)) 

    sampled_seqs=all_pos_seqs+neg_seqs_sample
    sampled_labels=all_pos_labels+[False]*len(neg_seqs_sample)

    return sampled_seqs, sampled_labels
    # pass

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
        # define encodings # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    code={'A':'1000', 'T':'0100', 'C':'0010', 'G':'0001'}
    code=dict((re.escape(k), v) for k, v in code.items()) 
    pattern=re.compile("|".join(code.keys()))

    # encode over loop and store in new list; goal is to replace each base pair with a binary string then convert the entire binary string to a list to create the encodings
    encodings=[]
    for seq in seq_arr:
        alt_seq=pattern.sub(lambda bp: code[re.escape(bp.group(0))], seq)
        alt_seq=list(alt_seq)
        alt_seq=list(map(float, alt_seq))
        encodings.append(alt_seq)

    return encodings
    # pass