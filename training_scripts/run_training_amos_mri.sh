#!/bin/sh

DATASET_PATH=../DATASET_AMOSMRI

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_amos_mri
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_preprocessed

python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_amos_mri 7 0
