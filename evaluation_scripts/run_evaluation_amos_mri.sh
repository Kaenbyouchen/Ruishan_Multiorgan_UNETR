#!/bin/sh

DATASET_PATH=../DATASET_AMOSMRI
CHECKPOINT_PATH=../output_amos_mri

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_preprocessed
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_amos_mri 7 0 -val
