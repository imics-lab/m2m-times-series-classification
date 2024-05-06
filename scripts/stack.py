#!/usr/bin/python3
"""
Sequence to vector (many-to-one) post processing.

Usage: stack.py [-v]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)

make sure cfg.py matches the data to be analyzed, it must be
the identical configuration from when the model was trained
this script works for both binary and categorical multi-label output

Stacking: Fully trained se2seq model is saved and used.
MUST have A/P for VALID file, and access to TEST
"""

# notes from meeting
# (use validation set) - Done
# will be a matrix of probabilities per label so flatten it [seqlen x outputs] - Done

###########################
# imports
###########################

# python libs
import os
import sys
import re
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

from os import getcwd
sys.path.insert(1, getcwd() + '/../src/')

# local libs
import cfg
import functions as fn

argc = len(sys.argv)
if argc >= 2 and sys.argv[1] == '-v':
    VERBOSE = True
else:
    VERBOSE = False

np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True)
if not os.path.exists(cfg.SAVEMODELPATH):
    raise FileNotFoundError("model not found.")
else:
    model = tf.keras.models.load_model(cfg.SAVEMODELPATH)

if VERBOSE:
    print("Model loaded...")
    model.summary()
    print()

# ##############################################################################
# Load or generate raw outputs from VALID,
# These will be the inputs to stacking model.
# Will be a matrix of seqlen x outputs, which is flattened for use in the model.
# ##############################################################################
if not os.path.exists(cfg.PREDICTRAWVALID):
    if VERBOSE: print("No validation predictions found, generating predictions...")

    inputs = fn.load_input_data(cfg.VALID, norm=False, outcols=cfg.OUTPUT, index=cfg.INDEX, header=cfg.HEADER)

    index = 0  # starting point of sliding window
    inputX = [] # holds input lists for complete seq2seq return data
    while index < inputs.shape[0] - cfg.SEQLENGTH + 1:  # stop when less than one sequence remains
        inputX.append(inputs[index: index + cfg.SEQLENGTH, :])
        index += 1
    inputX = np.array(inputX)

    if VERBOSE:
        print("Shapes loaded...")
        print("  pre shaping:", inputs.shape)
        print("  post shaping:", inputX.shape)
    del inputs

    if VERBOSE: print("Beginning prediction capture from validation file...")

    if VERBOSE:
        tenpercent = int(inputX.shape[0] * .1)
        percent = -10

    inputs = []
    for i in range(inputX.shape[0]):
        sequence = inputX[i]
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        yhat = model.predict(sequence)
        yhat = yhat.flatten()
        inputs.append(yhat)
        if VERBOSE and not (float(i) % tenpercent):
            percent += 10
            print(percent, "%...", sep="")

    inputs = np.array(inputs)

    if VERBOSE: print("Done.", inputs.shape[0], "predictions made for use as stacking input. Writing to file...")

    fout = open(cfg.PREDICTRAWVALID, "w")
    for sequence in inputs:
        line = str(sequence)[1:-1]
        line = re.sub("\s+", ",", line.strip())
        fout.write(line+"\n")
    fout.close()
    if VERBOSE: print("Done.")
else:
    if VERBOSE: print("Loading predictions for stacking input...", end="")
    fin = open(cfg.PREDICTRAWVALID, "r")
    inputs = []
    for line in fin:
        line = line[:-1].split(",")
        line = fn.text_to_numpy(line)
        inputs.append(line)
    fin.close()
    if VERBOSE: print("Done.")

    inputs = np.array(inputs)

# ##############################################################################
# Getting actual outputs from the VALID file
# These will be the outputs to stacking model.
# ##############################################################################
if VERBOSE: print("Loading validation output for training...", end="")

fin = open(cfg.VALID, "r")
if cfg.HEADER == 0:
    fin.readline() # throw away header
outputs = []
for line in fin:
    outputs.append(fn.text_to_numpy(line[:-1].split(",")[-cfg.OUTPUT:]))
fin.close()

# remove the first seqlen elements because a full sequence is needed for first prediction
outputs = np.array(outputs)[cfg.SEQLENGTH-1:]
if VERBOSE: print("Done.")

if VERBOSE:
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)
    print("Fitting Model...", end="")

# #####################################
# make and fit model to validation data
# inputs: the VALID file raw ouputs from the model
# outputs: the VALID file actual outputs
# stacking model is trained on VALID outputs used as input
# against actual outputs from VALID file
# #####################################
model = XGBClassifier()
model.fit(inputs, outputs)

if VERBOSE:
    print("Done.")
    print("Loading test data for testing...", end="")

# ###############################################################################
# Getting raw output created from test file after training seq2seq
# This is the output from seq2seq model testing for comparison to stacking output
# ###############################################################################
lines = fn.getFileLines(cfg.PREDICTRAW)
# sanity check
if lines % cfg.SEQLENGTH:
    raise ValueError("oops")
else:
    testinputs = lines // cfg.SEQLENGTH

fin = open(cfg.PREDICTRAW, "r")
test = []
sequence = np.zeros([cfg.OUTPUT*cfg.SEQLENGTH])
for i in range(testinputs):
    tmpstr = ""
    for j in range(cfg.SEQLENGTH):
        line = fin.readline()[:-1] + ","
        tmpstr += line
    test.append(fn.text_to_numpy(tmpstr[:-1].split(",")))
fin.close()
test = np.array(test)

# ######################################################################
# make predictions from test data using model trained on validation data
# load actual from TEST for comparison
# ######################################################################
y_pred = model.predict(test)

if VERBOSE: print("Loading test output for testing...", end="")

fin = open(cfg.TEST, "r")
if cfg.HEADER == 0:
    fin.readline() # throw away header
y_actual = []
for line in fin:
    y_actual.append(fn.text_to_numpy(line[:-1].split(",")[-cfg.OUTPUT:]))
fin.close()

# remove the first seqlen elements because a full sequence is needed for first prediction
y_actual = np.array(y_actual)[cfg.SEQLENGTH-1:]
if VERBOSE: print("Done.")

#np.savetxt('y_a1.csv', y_actual, delimiter=',')
#np.savetxt('y_p1.csv', y_pred, delimiter=',')

# evaluate predictions
accuracy = accuracy_score(y_actual, y_pred)
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))

# ###########################################
# define the new model with sklearn model
# ###########################################
model = GradientBoostingClassifier()

# turn onehot outputs into categorical for the model, because GradientBoostingClassifier requires it
outs = []
for vector in outputs:
    outs.append(fn.categorical(vector))
outs = np.array(outs)
model.fit(inputs, outs)

y_pred = model.predict(test)

# turn onehot actual into categorical for direct comparison in accuracy_score
y_act = []
for vector in y_actual:
    y_act.append(fn.categorical(vector))
y_act = np.array(y_act)

#np.savetxt('y_a2.csv', y_act, delimiter=',')
#np.savetxt('y_p2.csv', y_pred, delimiter=',')

# evaluate predictions
accuracy = accuracy_score(y_act, y_pred)
print("SciKit XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))
