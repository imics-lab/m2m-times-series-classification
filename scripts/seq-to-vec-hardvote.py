#!/usr/bin/python3
"""
Sequence to vector (many-to-one) post processing.

Usage: seq-to-vec-hardvote.py [-v]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)

make sure cfg.py matches the data to be analyzed, it must be
the identical configuration from when the model was trained
this script works for both binary and categorical multi-label output

file renamed from seq-to-vec.py to seq-to-vec-hardvote.py
"""

###########################
# imports
###########################

# python libs
import os
import sys
import silence_tensorflow.auto
from os import getcwd
sys.path.insert(1, getcwd() + '/../src/')
import numpy as np

# local libs
import cfg
import functions as fn
import scrfunctions as sfn

argc = len(sys.argv)
if argc >= 2 and sys.argv[1] == '-v':
    VERBOSE = True
else:
    VERBOSE = False

if not os.path.exists(cfg.ACTUALPREDICTEDSEQ):
    raise FileNotFoundError(os.path.basename(cfg.ACTUALPREDICTEDSEQ) + " not found.")

if VERBOSE:
    print("Processing file: '%s'" % os.path.basename(cfg.ACTUALPREDICTEDSEQ))

# ##############################
# geting total time-steps
# ##############################

if VERBOSE:
    print("Counting time-steps...", end=" ")
timesteps = fn.getFileLines(cfg.ACTUALPREDICTEDSEQ)
if VERBOSE:
    print(timesteps, "time-steps.")

# ##############################
# processing into vectors
# ##############################

# create arrays to hold actual and predicted
a_seq2seq = np.full((timesteps, cfg.SEQLENGTH), -1, dtype=float)
p_seq2seq = np.full((timesteps, cfg.SEQLENGTH), -1, dtype=float)

# split out actual and predicted seq2seq
apfin = open(cfg.ACTUALPREDICTEDSEQ, "r")
for i in range(timesteps):
    line = apfin.readline()[:-1].split(",")
    line = fn.text_to_numpy(line)
    a_seq2seq[i][:] = line[:cfg.SEQLENGTH]
    p_seq2seq[i][:] = line[cfg.SEQLENGTH:]
apfin.close()

actual = sfn.get_actual_vec(a_seq2seq)

if VERBOSE:
    print("Actual and predicted sequences loaded.")

# example pattern produced by the following code for seqlen 6
# i+t,j
# 0   5  time-step 0, 5th time-step prediction
# 1   4  time-step 1, 4th time-step prediction which lines up with previous 0,5
# 2   3  time-step 2, 3rd time-step prediction which lines up with previous 1,3
# 3   2  etc
# 4   1
# 5   0
# produces vector like this: [1, 1, 2, 2, 2, 3] which can be used to "vote"
# for the single prediction at time-step 5, i.e the first prediction in seq-to-vec model

# 1   5  time-step 1, 5th time-step prediction
# 2   4  time-step 2, 4th time-step prediction which lines up with previous 2,5
# 3   3  time-step 3, 3rd time-step prediction which lines up with previous 3,3
# 4   2  etc
# 5   1
# 6   0
# produces vector like this: [3, 3, 3, 3, 4, 4] which can be used to "vote"
# for the single prediction at time-step 6, i.e the second prediction in seq-to-vec model
# etc

if VERBOSE:
    print("Converting sequence to vector with hard vote...", end=" ")

fout = open(cfg.ACTUALPREDICTEDHARDVEC, "w")
vector = np.full(cfg.SEQLENGTH, -1) # for column of values across time
count = 0 # to count the number of actual versus predicted and also to be an index
for t in range(timesteps-cfg.SEQLENGTH+1):
    # get "column" of values, turn into a vector for voting
    j = cfg.SEQLENGTH - 1
    for i in range(cfg.SEQLENGTH):
        vector[i] = p_seq2seq[i+t,j]
        j -= 1
    predicted = sfn.hardvote(vector)
    fout.write(str(actual[count]) + "," + str(predicted)+"\n")
    count +=1
fout.close()
if VERBOSE:
    print("Done.\n")

if VERBOSE:
    if os.path.exists(cfg.ACTUALPREDICTEDV):
        try:
            print("seq2vec test data using seq2vec modeling...")
            fn.print_stats(cfg.ACTUALPREDICTEDV, cfg.OUTPUT)
        except:
            print("Cannot process", cfg.ACTUALPREDICTEDV, "Are you sure it's the right one?")
    print("seq2seq test data after post-processing to seq2vec with hard voting...")
    fn.print_stats(cfg.ACTUALPREDICTEDHARDVEC, cfg.OUTPUT)
