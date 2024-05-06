#!/usr/bin/python3
"""
Sequence to vector (many-to-one) soft vote post processing for multiple label output

Usage: seq-to-vec-softvote.py [-va]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)
a: use attention as weight in soft vote (optional)

make sure cfg.py matches the data to be analyzed, it must be
the identical configuration from when the model was trained
this script works ONLY for categorical multi-label output
"""

###########################
# imports
###########################

# python libs
import os
import sys
import re
import silence_tensorflow.auto
from os import getcwd
sys.path.insert(1, getcwd() + '/../src/')
import numpy as np

# local libs
import cfg
import functions as fn
import scrfunctions as sfn

# set defaults
VERBOSE = False
USEATTN = False
np.set_printoptions(precision=3, suppress=True)

# get command line arguments
argc = len(sys.argv)
if argc == 2 and re.search("^-[va]+", sys.argv[1]):
    if 'v' in sys.argv[1]:
        VERBOSE = True
    if 'a' in sys.argv[1]:
        USEATTN = True

if not os.path.exists(cfg.PREDICTRAW):
    raise FileNotFoundError(os.path.basename(cfg.PREDICTRAW) + " not found.")
elif not os.path.exists(cfg.ACTUALPREDICTEDS):
    raise FileNotFoundError(os.path.basename(cfg.ACTUALPREDICTEDS) + " not found.")
if USEATTN and not os.path.exists(cfg.TATTN):
    raise FileNotFoundError(os.path.basename(cfg.TATTN) + " not found.")

if VERBOSE:
    print("\nSequence length:", cfg.SEQLENGTH)
    print("Number of classes:", cfg.OUTPUT)
    print("Processing files: '", os.path.basename(cfg.PREDICTRAW), "' and '", os.path.basename(cfg.ACTUALPREDICTEDS),
          "'", sep="")
    if USEATTN:
        print("Using attention file '",os.path.basename(cfg.TATTN), "' in soft vote.", sep="")

# #########################################
# geting total time-steps and sanity checks
# #########################################

# get aplines and aptimesteps
aplines = fn.getFileLines(cfg.ACTUALPREDICTEDS)

# sanity check for actual / predicted file
if not aplines % cfg.SEQLENGTH:
    aptimesteps = aplines // cfg.SEQLENGTH
else:
    raise ValueError('lines do not divide evenly with sequences')

if VERBOSE:
    print("Actual / Predicted file contains...")
    print("  ", aplines, "lines\n  ", aptimesteps, "time-steps")

if USEATTN:
    # get atlines and attimesteps
    fin = open(cfg.TATTN, "r")
    atlines = 0
    while(fin.readline()):
        atlines += 1
    fin.close()

    # sanity check for attention
    if not atlines % cfg.SEQLENGTH:
        attimesteps = atlines // cfg.SEQLENGTH
    else:
        raise ValueError('attention file lines do not divide evenly with sequences')

    if VERBOSE:
        print("Attention file contains...")
        print("  ", atlines, "lines\n  ", attimesteps, "time-steps")

    # sanity check numbers match
    if aplines != atlines:
        raise ValueError('lines between actual/predicted and attention do not match')
    elif aptimesteps != attimesteps:
        raise ValueError('timesteps between actual/predicted and attention do not match')

# ##############################
# processing into vectors
# ##############################

# Get actual
# create array to hold actual for actual versus predicted comparison
a_seq2seq = np.full((aptimesteps, cfg.SEQLENGTH), -1, dtype=float)
apfin = open(cfg.ACTUALPREDICTEDS, "r")
count = 0
for i in range(aptimesteps):
    for j in range(cfg.SEQLENGTH):
        count += 1
        line = apfin.readline()[:-1].split(",")
        line = fn.text_to_numpy(line)
        a_seq2seq[i][j] = line[0]
apfin.close()
actual = sfn.get_actual_vec(a_seq2seq)

if VERBOSE:
    print("Total actual seq-to-vec predictions:", actual.shape[0], "(time-steps-seqlen-1)")
    print("Loading predicted... ", end="")

# Get predicted
# create array to hold predicted, each row is all output per one sequence
p_seq2seq = np.full((aptimesteps, cfg.SEQLENGTH*cfg.OUTPUT), -1, dtype=float)

fin = open(cfg.PREDICTRAW, "r")
for i in range(aptimesteps):
    for j in range(cfg.SEQLENGTH):
        line = fn.text_to_numpy(fin.readline()[:-1].split(","))
        for k in range(cfg.OUTPUT):
            p_seq2seq[i][j*cfg.OUTPUT+k] = line[k]
fin.close()

if VERBOSE:
    print(p_seq2seq.shape[0], " X (", p_seq2seq.shape[1]//cfg.OUTPUT, "x", cfg.OUTPUT, ") loaded.", sep="")

if VERBOSE:
    if USEATTN:
        print("Converting sequence to vector with soft vote using attention...", end=" ")
    else:
        print("Converting sequence to vector with soft vote...", end=" ")

if USEATTN:
    attn = np.full((aplines), -1, dtype=float)
    fin = open(cfg.TATTN, "r")
    for i in range(aplines):
        attn[i] = float(fin.readline()[:-1])
    attn = attn.reshape([aptimesteps,cfg.SEQLENGTH])

fout = open(cfg.ACTUALPREDICTEDSOFTVEC, "w")
matrix = np.full([cfg.SEQLENGTH, cfg.OUTPUT], -1, dtype="float")  # for column of values across time
for t in range(aptimesteps-cfg.SEQLENGTH+1):
    # get matrix of values (seqlen X seqlen(x)labels
    wholematrix = p_seq2seq[t:t+cfg.SEQLENGTH,:]

    # get seqlen x label for voting
    for i in range(cfg.SEQLENGTH):
        matrix[i] = wholematrix[i][(cfg.SEQLENGTH-1-i)*cfg.OUTPUT : ((cfg.SEQLENGTH-1-i)*cfg.OUTPUT)+cfg.OUTPUT]
    if USEATTN:
        matrix = matrix * attn[t][:, np.newaxis]
    predicted = np.argmax(np.sum(matrix, axis=0))
    fout.write(str(actual[t]) + "," + str(predicted)+"\n")
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
    if USEATTN: attnstr = "attention "
    else: attnstr = ""
    print("seq2seq test data after post-processing to seq2vec with soft %svoting..." % attnstr)
    fn.print_stats(cfg.ACTUALPREDICTEDSOFTVEC, cfg.OUTPUT)
