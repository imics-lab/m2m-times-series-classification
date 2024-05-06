"""
Supporting stand-alone functions for scripts
This file does not run on it's own, it's supporting functions only.

"""
import os
import sys
import numpy as np
from os import getcwd

sys.path.insert(1, getcwd() + '/../src/')
import cfg

def get_actual_vec(aseqs):
    """
    converts seq2seq actual values back into the vector from which they came

    @param (numpy 2D) aseqs : seq2seq actual values

    Return: seq2vec actual values (matches what's in the original data file
    """

    length = aseqs.shape[0]-cfg.SEQLENGTH+1
    avec = np.full((length), -1)
    for i in range(length):
        avec[i] = aseqs[i][cfg.SEQLENGTH-1]
    return avec

def hardvote(vec):
    """
    takes a hard vote from each time-step's prediction for a particular time-step

    @param (numpy 1D) vec : pre-processed vector of the column of predictions

    Return: the most common value which is the "vote" of all the time-steps
    """

    # TODO what if they are equal?
    # ignore it since it's a rare case and soft voting solves it?

    # get values and corresponding counts
    values, counts = np.unique(vec, return_counts=True)
    most_common_value = values[np.argmax(counts)]
    #print(i+1, vec)
    #print(counts, values)
    #print(most_common_value)
    return most_common_value

