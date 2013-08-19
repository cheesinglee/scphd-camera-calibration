#!/bin/bash

rm *.mat

# DATA=/home/cheesinglee/ownCloud/workspace/scphd-camera-calibration/data/disparity_sim.mat
CONFIG=./config/calibration.cfg
N_RUNS=100

RESULTS_DIR=./results
cp $CONFIG $RESULTS_DIR
for ((n=1; n <= $N_RUNS; n++))
do
    echo $n
    DATA=./data/batch/$n/data.mat
    ./scphd_camera_calibration $CONFIG $DATA | tee stdout.log
    mkdir -p $RESULTS_DIR/${n}
    mv *.mat $RESULTS_DIR/${n}
    mv stdout.log $RESULTS_DIR/${n}
done

