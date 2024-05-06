#!/usr/bin/python3
"""
Simple stand alone analysis.

Usage: analysis.py filename labels
       (assuming python3 in /usr/bin/)

file input must conform to cfg.ACTUALPREDICTEDV or cfg.ACTUALPREDICTEDS
labels must be the number of categorical outputs

WARNING: script has few sanity checks. Make sure you have the right file and labels

"""

###########################
# imports
###########################

# python libs
import os
import sys
import silence_tensorflow.auto
sys.path.insert(1, os.getcwd() + '/../src/')

# local libs
import functions as fn

argc = len(sys.argv)
labels = 0
if argc == 3 and os.path.exists(sys.argv[1]):
    try:
        labels = int(sys.argv[2])
    except:
        print("You didn't supply a proper integer for number of labels")
else:
    print("You didn't supply either a correct filename or proper integer for number of labels")

if labels:
    fn.print_stats(sys.argv[1], labels)
