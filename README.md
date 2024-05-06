# Effective Modeling of Frequent Label Transitions in Time Series

## Description
Code from the paper titled "Effective Modeling of Frequent Label
Transitions in Time Series" in review.

## Abstract.
Time series classification is vital in areas
like financial forecasting, health monitoring,
and human activity recognition. Traditional
time-series models segment data into win-
dows and assign one label per window, often
missing label transitions within those win-
dows. This paper presents a many-to-many
time-series model using hybrid recurrent neu-
ral networks with attention mechanisms. Un-
like typical many-to-many models, our ap-
proach doesn’t require a decoder. Instead,
it employs an RNN generating a label for
every input time step. During inference, a
weighted voting scheme consolidates overlap-
ping predictions into one label per time step.
Experiments show our model remains effec-
tive on time series with sparse label shifts,
but particularly excels in detecting frequent
transitions. This model is ideal for tasks de-
manding accurate pinpointing of rapid label
changes in time-series data.

## Setup
Requires
 - Python 3.6 or greater
 - TensorFlow 2.4 or greater
 - Compatible versions of
   - numpy
   - pandas
   - xgboost
   - sklearn
   - matplotlib

## Usage
Usage: main.py [-vg]
(assuming python3 in /usr/bin/)

v: verbose mode (optional)
g: graphing mode (optional)

Configuration settings in src/cfg.py

 - run training first with SEC2SEQ=False to get A/P many-to-one for analysis and comparison to many-to-many
 - then run this file with cfg.SEC2SEQ=True to get A/P many-to-many for analysis and comparison
 - for many-to-one to many-to-many analysis and comparison
    - run hard vote
    - run soft vote without attention
    - run soft vote with attention
    - run stacking
    - compare all five

## Citation
See paper Effective Modeling of Frequent Label Transitions in Time Series
