"""
Supporting stand-alone functions for main

"""

import re
import sys
import os
import glob
import random
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import tensorflow as tf
import silence_tensorflow.auto
import cfg

def get_args():
    """
    Get command line arguments at program start.
    Return: the set flags: verbose, graph

    Usage: main.py [-vg]
           (assuming python3 in /usr/bin/)
    v: verbose mode (optional)
    g: graphing mode (optional)

    """

    # set optional defaults in case of error or no parameters
    verbose = cfg.VERBOSE
    graph = cfg.GRAPH

    if len(sys.argv) == 2 and re.search("^-[vg]+", sys.argv[1]):
        if 'v' in sys.argv[1]:
            verbose = True
        if 'g' in sys.argv[1]:
            graph = True

    return verbose, graph


def print_setup(v, g):
    """
    Print runtime setup information

    @param (bool) v : verbose mode on/off
    @param (bool) g : graphing error curves on/off

    Return: nothing
    """
    print("TF Version:", tf. __version__)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install the GPU version of TF")
    print()
    if cfg.SEC2SEQ:
        print("Executing many-to-many model.")
    else:
        print("Executing many-to-one model.")
    if cfg.SHUFFLE:
        print("Sequences shuffled.")
    else:
        print("Sequences NOT shuffled.")
    if cfg.EAGERLY:
        print("Running eagerly.")
    if cfg.CAPATTN:
        print("Capturing attention.")
    else:
        print("NOT capturing attention.")
    if cfg.SAVEMODEL:
        print("Model will be saved.")
    else:
        print("Model will NOT be saved.")
    print(" - Verbose: On") if v else print(" - Verbose: Off")
    print(" - Graphing: On") if g else print(" - Graphing: Off")
    print()


def text_to_numpy(line, type="float"):
    """
    turn a comma separated line of numbers into a numpy array

    @param (string) line : comma separated line of numbers
    @param (string) type : int or float

    Return: numpy array
    """

    if type != "int" and type != "float":
        type = "float"

    length = len(line)
    nparray = np.empty((length), dtype=type)
    for i, n in enumerate(line):
        if type=="int":
            nparray[i] = int(line[i])
        else:
            nparray[i] = float(line[i])
    return nparray


def numpy_to_text(array, delim=","):
    """
    turn a numpy array into a comma separated line of numbers (string)

    @param (numpy array) array : comma separated line of numbers
    @param (string) delim : delimiter

    Return: string
    """

    length = array.shape[0]
    array = array.tolist()
    for i in range(length):
        array[i] = str(array[i])
    return delim.join(array)


def getFileLines(filename):
    """
    get the number of lines in a file

    @param (string) filename : comma separated line of numbers

    Return: int (count of the lines)
    """
    with open(filename, 'r') as fin  :
        lines = 0
        while fin.readline():
            lines += 1
    return lines


def create_attn_capure_files():
    """
    create empty attention capture files
    for use in predictCb
    bad design, but it works
    """
    fout = open(cfg.TATTN, "w")
    fout.close()
    fout = open(cfg.TATTN_NORM, "w")
    fout.close()


def delete_ap_files():
    """
    delete actual and predicted files
    bad design, but it works

    dont delete cfg.ACTUALPREDICTEDS
    """

    if os.path.exists(cfg.PREDICTRAW):
        os.remove(cfg.PREDICTRAW)
    if os.path.exists(cfg.ACTUALPREDICTEDS):
        os.remove(cfg.ACTUALPREDICTEDS)
    if os.path.exists(cfg.ACTUALPREDICTEDSEQ):
        os.remove(cfg.ACTUALPREDICTEDSEQ)
    if os.path.exists(cfg.ACTUALPREDICTEDHARDVEC):
        os.remove(cfg.ACTUALPREDICTEDHARDVEC)
    if os.path.exists(cfg.ACTUALPREDICTEDSOFTVEC):
        os.remove(cfg.ACTUALPREDICTEDSOFTVEC)
    if os.path.exists(cfg.ACTUALPREDICTEDATTNVEC):
        os.remove(cfg.ACTUALPREDICTEDATTNVEC)
    if os.path.exists(cfg.PREDICTRAWVALID):
        os.remove(cfg.PREDICTRAWVALID)
    for f in glob.glob(cfg.RESULTSPATH + "cm*.csv"):
        os.remove(f)

def delete_attn_capure_files():
    """
    delete attention capture files
    again, bad design, but it works and life is hard
    """

    if os.path.exists(cfg.TATTN):
        os.remove(cfg.TATTN)
    if os.path.exists(cfg.TATTN_NORM):
        os.remove(cfg.TATTN_NORM)
    if os.path.exists(cfg.TATTN_NORM_PROCESSED):
        os.remove(cfg.TATTN_NORM_PROCESSED)
    if os.path.exists(cfg.TATTN_SIGNATURES):
        os.remove(cfg.TATTN_SIGNATURES)

def print_stats(filename, label_count=2, save_cm="", graph=True):
    """
    accept an actual/predicted file and print out classification_report

    @param (string) filename : actual/predicted file
    @param (int) label_count : number of labels, defaults to one actual, one predicted

    """

    if label_count == 1:
        label_count = 2
    lines = getFileLines(filename)

    y_true = np.zeros([lines], dtype=float)
    y_pred = np.zeros([lines], dtype=float)

    fin = open(filename, "r")
    i = 0
    for line in fin:
        line = line[:-1].split(",")
        y_true[i] = int(float(line[0]))
        y_pred[i] = int(float(line[1]))
        i += 1
    fin.close()

    target_names = []
    for i in range(label_count):
        target_names.append(str(i))

    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    cmn = confusion_matrix(y_true, y_pred, normalize = 'all')
    if save_cm != "":
        np.savetxt(cfg.RESULTSPATH + 'cm_' + save_cm + ".csv", cm, delimiter=",")
        np.savetxt(cfg.RESULTSPATH + 'cmn_' + save_cm + ".csv", cm, delimiter=",")
    print(cm)
    print(cmn)

    if graph:
        x_axis = np.arange(y_true.shape[0])

        y_true += .020
        y_pred -= .020

        pyplot.plot(x_axis, y_true, color='red')
        pyplot.plot(x_axis, y_pred, color='blue')
        pyplot.yticks(np.arange(label_count))
        pyplot.show()
    return


def transitions(y):
    """
    count the number of sequences which contain a transition from one label to another

    @param (3d numpy array) y : 3d sequenes of outputs for use in seq2seq models

    Return: int, float (count of transitions, percent of total)
    """

    nsequences = y.shape[0]
    seqlen = y.shape[1]

    ntransition_seqs = 0
    for i in range(nsequences):
        first_label = categorical(y[i][0])
        transition = False
        for j in range(1, seqlen):
            label = categorical(y[i][j])
            if first_label != label:
                transition = True
                break
        if transition:
            ntransition_seqs += 1

    return [ntransition_seqs, round(float(ntransition_seqs) / float(nsequences),2)]


def onehotencode(categorical, totalcategories):
    """
    one-hot encoding
    example, assumes 100 = 0, 010 = 1, 001 = 3
    assumes categories start at 0 and 1 step to totalcategories-1

    @param (int) categorical : a single categorical value
    @param (int) totalcategories : a count of the number of total categories

    Return: one hot encoded vector
    """

    #sanity checks
    if categorical >= totalcategories:
        raise ValueError("category cannot be larger than total categories -1 ")
    elif categorical < 0:
        raise ValueError("category can not be less than 0")
    elif totalcategories < 1:
        raise ValueError("total categories can be less than 1")

    onehot = np.zeros([totalcategories])
    onehot[categorical] = 1
    return onehot


def categorical(outputs):
    """
    reverses one-hot encoding
    example, assumes 100 = 0, 010 = 1, 001 = 3

    @param (nparray) outputs : an vector of one hot encoded ints

    Return: categorical label starting with 0
    """

    n = len(outputs)
    category = -1
    for i in range(n):
        if outputs[i] == 1.0: category = i
    return category


def load_input_data(filename, outcols=1, norm=False, index=None, header=None):
    """
    Load only a dataset's inputs (features) from csv. File is assumed to be in the form
    timesteps (rows) X features + labels (columns). All features are assumed to
    be before all labels (i.e. labels are the last columns)

    @param (string) filename : file name
    @param (int) outcols : number of columns of output labels, assumes at right most
    @param (bool) norm : normalize the data
    @param (int) index : presence of an index (none or column number)
    @param (int) header : presence of aheader (none or row number)

    Return: datasets as numpy arrays
    """

    inputs = pd.read_csv(filename, index_col=index, header=header).astype(float, errors='ignore')
    features = inputs.shape[1] - outcols

    inputs = inputs.iloc[:, 0:features]

    if norm:
        [inputs[col].update((inputs[col] - inputs[col].min()) / (inputs[col].max() - inputs[col].min())) for col in
         inputs.columns]

    return inputs.to_numpy()


def load_data(train_name, valid_name, test_name="", outcols=1, norm=False, index=None, header=None):
    """
    Load a dataset from csv. File is assumed to be in the form
    timesteps (rows) X features + labels (columns). All features are assumed to
    be before all labels (i.e. labels are the last columns)

    @param (string) train_name : training file name
    @param (string) valid_name : validation file name (optional)
    @param (string) test_name : test file name
    @param (int) outcols : number of columns of output labels, assumes at right most
    @param (bool) norm : normalize the data
    @param (int) index : presence of an index (none or column number)
    @param (int) header : presence of aheader (none or row number)

    Return: datasets as numpy arrays
    """

    train = pd.read_csv(train_name, index_col=index, header=header).astype(float, errors='ignore')
    valid = pd.read_csv(valid_name, index_col=index, header=header).astype(float, errors='ignore')

    features = train.shape[1] - outcols

    train_x = train.iloc[:, 0:features]
    train_y = train.iloc[:, features:]
    del train

    valid_x = valid.iloc[:, 0:features]
    valid_y = valid.iloc[:, features:]
    del valid

    if test_name:
        test = pd.read_csv(test_name, index_col=index, header=header).astype(float, errors='ignore')
        test_x = test.iloc[:, 0:features]
        test_y = test.iloc[:, features:]
        del test
    else:
        test_x = valid_x.copy()
        test_y = valid_y.copy()

    if norm:
        [train_x[col].update((train_x[col] - train_x[col].min()) / (train_x[col].max() - train_x[col].min())) for col in
         train_x.columns]
        [valid_x[col].update((valid_x[col] - valid_x[col].min()) / (valid_x[col].max() - valid_x[col].min())) for col in
         valid_x.columns]
        [test_x[col].update((test_x[col] - test_x[col].min()) / (test_x[col].max() - test_x[col].min())) for col in
         test_x.columns]

    return train_x.to_numpy(), train_y.to_numpy(),\
           valid_x.to_numpy(), valid_y.to_numpy(),\
           test_x.to_numpy(), test_y.to_numpy()


def save3d_to_csv(data, filename, format='%f'):
    """
    save 3D data (seqeunced data) to csv

    @param (3D Numpy) data : data to be saved
    @param (string) filename : csv file name inlcuding location
    @param (string) format : integer or float (default)

    Return: nothing
    """

    # sanity check
    if format != '%f' or format != '%d':
        format = '%f'
    if data.ndim==3:
        # reshape to 2D
        data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    np.savetxt(filename, data, delimiter=',', fmt=format)
    return


def shape_3D_data(data, seqlen):
    """
    Reshape 2D data into 3D data of groups of 2D time-steps or sequence length

    @param (2D Numpy) data : data to be reshaped
    @param (int) seqlen : sequence length

    Return: Reshaped data as 3D numpy array
    """
    if len(data.shape) != 2:
        raise TypeError("Input data must be 2D.")

    # samples are total number of input vectors
    samples = data.shape[0]
    # features are number of columns
    features = data.shape[1]

    # samples must divide evenly by seqlen to create an even set of batches
    # if they don't, drop samples off the end to make it even
    if samples % seqlen:
        remove = samples - ((samples // seqlen) * seqlen)
        data = np.delete(data, slice(samples - remove, samples), 0)

    return data.reshape(data.shape[0] // seqlen, seqlen, features)


def shape_slide_3D_data(dataX, dataY, seqlen, randmax=0, slide=0, seq2seq=True):
    """
    Reshape 2D data into 3D data with a fixed or randomly sliding window

    @param (2D Numpy) dataX : input data to be reshaped
    @param (2D Numpy) dataY : output data to be reshaped
    @param (int) seqlen : sequence length
    @param (int) randmax : maximum distance of random slide, default's to sequence length (-1)
    @param (int) slide : size of the slide, 0 = random from 1 to sequence length

    Return: Reshaped dataX and dataY as 3D numpy arrays

    Note: unlike shape_3D_data(), X and Y must be done together to use same random numbers
    """

    # sanity checks
    if dataX.shape[0] != dataY.shape[0]:
        raise ValueError("X,Y instances don't match.")
    if len(dataX.shape) != 2 and len(dataY.shape) != 2:
        raise TypeError("Input data must be 2D.")
    if slide < 0 or slide > seqlen:
        slide = seqlen # default to no overlap, same as shape_3D_data()
    if randmax > seqlen or randmax < 1:
        randmax = seqlen

    # set up data
    index = 0 # starting point of sliding window
    retDataX = [] # holds input lists for complete seq2seq return data
    retDataY = [] # holds output lists for complete seq2seq or seq2vec return data
    if not slide: randomnums = []  # holds random slides in case we need them after processing
    else: randomnums = [slide]

    # get sequences
    while index < dataX.shape[0]-seqlen+1: # stop when less than one sequence remains
        # getting input is always the same
        retDataX.append(dataX[index : index + seqlen, :])

        # getting output is difference depending on seq2seq or sec2vec
        if seq2seq:
            # seq2sec case for both rnd and fixed slide
            retDataY.append(dataY[index: index + seqlen, :])
        else:
            # seq2vec case
            if slide == 0:
                # random sliding window
                # will append the mode of the sequence output, if there is more than one mode,
                # it will use the last one, making it closer to the next theoretical transition

                # matrix of onehot time-steps
                sequenceY = dataY[index: index + seqlen, :]
                sequenceYcat = []
                # get vector of categories representing onehot matrix of time-steps
                for i in range(sequenceY.shape[1]):
                    sequenceYcat.append(categorical(sequenceY[i]))
                sequenceYcat = np.array(sequenceYcat)
                # get mode and turn back into onehot
                modes = statistics.multimode(sequenceYcat)
                onehotmode = onehotencode(modes[len(modes) - 1],cfg.OUTPUT)
                # save to return output data
                retDataY.append(onehotmode)
            elif slide > 1:
                # fixed window > 1
                # this is not implemented yet because it's an unneeded case at the moment
                raise NotImplementedError("Not implemented. What are you doing here?")

        # advance the window
        if not slide: # random case
            # slides the window from 1 to randmax (inclusive)
            r = random.randint(1, randmax)
            index += r
            randomnums.append(r) # save them in case we want to inspect them
        else:
            index += slide

    # special case, requires no looping, for seq-to-vec which will be not seq2seq and slide=1 automatically
    if not seq2seq and slide == 1:
        retDataY = dataY[cfg.SEQLENGTH - 1:, :]

    return np.array(retDataX), np.array(retDataY), np.array(randomnums)

def calculate_weights(Y):

    # Step 1: Identify unique transitions for each example
    Y_train_argmax = np.argmax(Y, axis=-1)
    transitions = np.diff(Y_train_argmax, axis=1)
    unique_transitions_per_example = [set(zip(row[:-1], row[1:])) for row in Y_train_argmax]

    # Step 2: Compute the transition frequencies across the dataset
    unique_transitions, transition_counts = np.unique(np.array([(a, b) for transitions in unique_transitions_per_example for a, b in transitions]), return_counts=True, axis=0)
    transition_frequencies_dict = dict(zip([tuple(t) for t in unique_transitions], transition_counts))

    # Step 3: Calculate the inverse of transition frequencies
    transition_weights_per_example = [np.sum([1 / transition_frequencies_dict[t] for t in transition_set]) for transition_set in unique_transitions_per_example]

    # Step 4: Normalize the weights so that they sum to 1
    transition_weights = np.array(transition_weights_per_example)
    transition_weights = transition_weights / np.sum(transition_weights)

    # Step 5: Assign the sample weights for each example based on the transitions
    sample_weights = transition_weights
    return sample_weights

# ######################################
# Functions for generating data
# ######################################

def generate_sequence(num_time_steps, num_features, label):
    means = {
        0: np.arange(0, num_features),
        1: np.arange(num_features, num_features * 2),
        2: np.arange(num_features * 2, num_features * 3)
    }
    return np.random.normal(loc=means[label], scale=1, size=(num_time_steps, num_features))

def generate_instance(num_time_steps, num_features, instance_type):
    if instance_type == 1:
        label = random.choice([0, 1, 2])
        return generate_sequence(num_time_steps, num_features, label), [label] * num_time_steps
    elif instance_type == 2:
        transition_point = random.randint(1, num_time_steps-1)
        label1 = random.choice([0, 1, 2])
        label2 = random.choice([x for x in [0, 1, 2] if x != label1])
        seq1, labels1 = generate_sequence(transition_point, num_features, label1), [label1] * transition_point
        seq2, labels2 = generate_sequence(num_time_steps - transition_point, num_features, label2), [label2] * (num_time_steps - transition_point)
        return np.concatenate((seq1, seq2)), labels1 + labels2
    else:
        point1 = random.randint(1, num_time_steps-2)
        point2 = random.randint(point1 + 1, num_time_steps-1)
        labels = [0, 1, 2]
        random.shuffle(labels)
        seq1, labels1 = generate_sequence(point1, num_features, labels[0]), [labels[0]] * point1
        seq2, labels2 = generate_sequence(point2 - point1, num_features, labels[1]), [labels[1]] * (point2 - point1)
        seq3, labels3 = generate_sequence(num_time_steps - point2, num_features, labels[2]), [labels[2]] * (num_time_steps - point2)
        return np.concatenate((seq1, seq2, seq3)), labels1 + labels2 + labels3

def generate_dataset(num_instances, num_time_steps, num_features):
    instances, labels = [], []
    for _ in range(num_instances):
        instance_type = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
        instance, instance_labels = generate_instance(num_time_steps, num_features, instance_type)
        instances.append(instance)
        labels.append(instance_labels)
    return np.array(instances), np.array(labels)

def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((*labels.shape, num_classes))
    for i, seq in enumerate(labels):
        for j, label in enumerate(seq):
            one_hot_labels[i, j, label] = 1
    return one_hot_labels

def generate_3D_data(num_features = 10, num_classes = 3, sequence_lenghth = 50, num_instances = 1000, verbose=False):

    instances, labels = generate_dataset(num_instances, sequence_lenghth, num_features)
    if verbose:
        print("Generated inputs:", instances.shape)  # Should be (1000, 50, 10)
        print("Generated labels:", labels.shape)  # Should be (1000, 50)

    one_hot_labels = one_hot_encode(labels, num_classes)
    if verbose:
        print("1-hot labels:", one_hot_labels.shape)  # Should be (1000, 50, 3)
        print()

    X = instances
    y = one_hot_labels

    # Split into train and valid_test
    train_X, valid_test_X, train_Y, valid_test_Y = train_test_split(X, y, train_size=.70)

    test_X, valid_X = np.split(valid_test_X, 2, axis=0)  # Array, num of splits, axis for splitting

    if verbose:
        print(test_X.shape, valid_X.shape)
        print()

    # Split valid_test into valid and test
    test_Y, valid_Y = np.split(valid_test_Y, 2, axis=0)  # Array, num of splits, axis for splitting
    del valid_test_X
    del valid_test_Y
    if verbose:
        print(test_Y.shape, valid_Y.shape)
    print()

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y
