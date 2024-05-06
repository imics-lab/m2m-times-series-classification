"""
Run Time Configuration
Many-to-one and Many-to-many modeling

"""

from os import getcwd

# ###############################
# data configuration
# ###############################
DATAPATH = getcwd() + '/../data/'
RESULTSPATH = getcwd() + '/../results/'
SAVEMODELPATH = RESULTSPATH + 'model/'
SAVESIGSPATH = RESULTSPATH + 'sigs/'


# -1 generated in code (no files)
#  0 fake (for code debug)
#  1 gesture multi-label classification
#  2 air quality binary label prediction
#  3 traffic binary prediction
#  4 traffic multi-label classification
DATA = 2

# #################################
# Most commonly changed
# Usually have to only change these
# #################################
SEC2SEQ = False
EPOCHS = 450
PATIENCE = 60

if DATA == -1:
    # data is generated on the fly in the code
    # for debuging only

    TRAIN = ""
    VALID = ""
    TEST = ""
    OUTPUT = 3
    INDEX = None
    HEADER = None
    SEQLENGTH = 50
    LSTM = 32
    RMAX = 1   # set this to 1, 2 or 3 to be full (smallest resuting set), 1/2, or 1/3 (largest resuting set) the max of the sequence slide  # set this to 1, 2 or 3 to be full (smallest resuting set), 1/2, or 1/3 (largest resuting set) the max of the sequence slide
elif DATA == 0:
    # need a good synthetic set here

    TRAIN = DATAPATH + "fake_train.csv"
    VALID = DATAPATH + "fake_valid.csv"
    TEST = DATAPATH + "fake_test.csv"
    OUTPUT = 2
    INDEX = None
    HEADER = None
    SEQLENGTH = 3
    LSTM = 8
    RMAX = 1   # set this to 1, 2 or 3 to be: 1/1 (largest slide, smallest resulting set), 1/2, or 1/3 (smallest slide, largest resulting set) the max of the sequence slide
elif DATA == 1:
    # https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation
    # multi-label

    # note:
    #  - for seq2vec, step=1, suffle=1 works best
    #  - for seq2seq, step=0, suffle=1 works best
    TRAIN = DATAPATH + "gesture/gtrain_ab13_82_1hm3G.csv"
    VALID = DATAPATH + "gesture/gvalid_ab13_82_1hm3G.csv"
    TEST = DATAPATH + "gesture/gtest_c13_1hm3G.csv"
    #TEST = DATAPATH + "gesture/gtest_a2_1hm3G.csv"
    OUTPUT = 3
    INDEX = None
    HEADER = None
    SEQLENGTH = 44
    LSTM = 128
    RMAX = 3   # set this to 1, 2 or 3 to be: 1/1 (largest slide, smallest resulting set), 1/2, or 1/3 (smallest slide, largest resulting set) the max of the sequence slide
elif DATA == 2:
    # https://archive.ics.uci.edu/ml/datasets/air+quality
    # binary-label

    TRAIN = DATAPATH + "aq_train-repaired.csv"
    VALID = DATAPATH + "aq_valid-repaired.csv"
    TEST = DATAPATH + "aq_test-repaired.csv"
    OUTPUT = 2
    INDEX = 0
    HEADER = 0
    SEQLENGTH = 8
    LSTM = 256
    RMAX = 2  # set this to 1, 2 or 3 to be: 1/1 (largest slide, smallest resulting set), 1/2, or 1/3 (smallest slide, largest resulting set) the max of the sequence slide
elif DATA == 3:
    # https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
    # bin, predict if the next hour traffic will (10) not increase or (01) increase

    TRAIN = DATAPATH + "traffic_processed2_train_b_1h.csv"
    VALID = DATAPATH + "traffic_processed2_valid_b_1h.csv"
    TEST = DATAPATH + "traffic_processed2_test_b_1h.csv"
    OUTPUT = 2
    INDEX = None
    HEADER = None
    SEQLENGTH = 12
    LSTM = 128
    RMAX = 2  # set this to 1, 2 or 3 to be: 1/1 (largest slide, smallest resulting set), 1/2, or 1/3 (smallest slide, largest resulting set) the max of the sequence slide
elif DATA == 4:
    # https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
    # multi-label classification

    TRAIN = DATAPATH + "traffic_processed2_train_m_1h.csv"
    VALID = DATAPATH + "traffic_processed2_valid_m_1h.csv"
    TEST = DATAPATH + "traffic_processed2_test_m_1h.csv"
    OUTPUT = 5
    INDEX = None
    HEADER = None
    SEQLENGTH = 12
    LSTM = 256
    RMAX = 2  # set this to 1, 2 or 3 to be: 1/1 (largest slide, smallest resulting set), 1/2, or 1/3 (smallest slide, largest resulting set) the max of the sequence slide
else:
    raise ValueError("data file error")

# ###############################
# command line parameter defaults
# ###############################
VERBOSE = True
GRAPH = False

# ###############################
# Hyperparameters
# ###############################

# STEP = 0 --> random slide, use for train and test, valid is hardcoded to always be 1
# STEP = 1 --> overlap of seqlen-1 (like time generator), valid is hardcoded to always be 1
# currently not using an step other than 0 and 1
STEP = 0 if SEC2SEQ else 1 # do not set this manually, use SEC2SEQ
WEIGHTED = True if SEC2SEQ else False # do not set this manually, use SEC2SEQ
SHUFFLE = True
BIDIRECTIONAL = True
DROPOUT = .25

# the following set automatically based on most common best performance
if BIDIRECTIONAL: DENSE1 = LSTM * 2
else: DENSE1 = LSTM
DENSE2 = DENSE1 / 2
# capture attention: 0 = no, 1 = general, 2 = self
# usually you want this setting, change it if needed
CAPATTN = 1 if SEC2SEQ else False
EAGERLY = True if CAPATTN else False # do not set this manually, use CAPATTN
# do not set this manually, use CAPATTN and SEC2SEQ, only need this for seq2seq sa voting
SAVEMODEL = True if CAPATTN and SEC2SEQ else False

# ###############################
# Data
# ###############################

# slide means there will be overlap of size step (the common case)
SLIDE = True # currently this should only be set to False when running fixed sequence data like ECG
RSAVE = False # save the shaped data for debug
BINTHRESHOLD = 0.5 # determines how soft output is converted to one hot
PREDICT = True


# ###############################
# Files
# ###############################

# save raw output from TEST, one complete set of cfg.OUTPUT per line
# needed in seq2seq for soft, attention voting, stacking
# not written for seq2vec modeling prediction
PREDICTRAW = RESULTSPATH + "predicted_raw.csv"

# save raw output from validation file, one complete sequence per line
# written and used by stack.py
PREDICTRAWVALID = RESULTSPATH + "predicted_raw_valid.csv"

# actual versus predicted, only one of these two is written during training and prediction
# ACTUALPREDICTEDV  is not deleted so it can be kept for analysis against seq2seq ACTUALPREDICTEDS
# if seq2vec, each line is a single point actual and prediction
ACTUALPREDICTEDV = RESULTSPATH + "actual_predicted_v.csv"
# if seq2seq, entire sequence is written, one actual and prediction value per line
ACTUALPREDICTEDS = RESULTSPATH + "actual_predicted_s.csv"

# ACTUALPREDICTED converted to one sequence of actual and one sequence of prediction per line
# used by hard vote as input
ACTUALPREDICTEDSEQ = RESULTSPATH + "actual_predicted_seq.csv"

# output of seq2seq conversions using hard-vote, soft-vote, attn-vote, etc.
ACTUALPREDICTEDHARDVEC = RESULTSPATH + "actual_predicted_hard_vec.csv"
ACTUALPREDICTEDSOFTVEC = RESULTSPATH + "actual_predicted_soft_vec.csv"
ACTUALPREDICTEDATTNVEC = RESULTSPATH + "actual_predicted_attn_vec.csv"

TATTN = RESULTSPATH + "tattn.csv" # all general attention layer activations
TATTN_NORM = RESULTSPATH + "tattn_norm.csv" # normalization of vattend.csv on a per seq basis
TATTN_NORM_PROCESSED = RESULTSPATH + "tattn_norm_processed.csv" # normalized sequences writen out by sequence
TATTN_SIGNATURES = RESULTSPATH + "tattn_signatures.csv"

if RSAVE:
    TRAIN_RNS = RESULTSPATH + "trains_rns.csv"
    VALID_RNS = RESULTSPATH + "valid_rns.csv"
    TEST_RNS = RESULTSPATH + "test_rns.csv"
    TRAIN_X_RESHAPED = RESULTSPATH + "train_X_reshaped.csv"
    TRAIN_Y_RESHAPED = RESULTSPATH + "train_Y_reshaped.csv"
    VALID_X_RESHAPED = RESULTSPATH + "valid_X_reshaped.csv"
    VALID_Y_RESHAPED = RESULTSPATH + "valid_Y_reshaped.csv"
    TEST_X_RESHAPED = RESULTSPATH + "test_X_reshaped.csv"
    TEST_Y_RESHAPED = RESULTSPATH + "test_Y_reshaped.csv"
