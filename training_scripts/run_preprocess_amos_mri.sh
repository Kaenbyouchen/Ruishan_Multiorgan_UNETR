#!/bin/sh

DATASET_PATH=../DATASET_AMOSMRI
AMOS_ROOT=../../amos22

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_amos_mri
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_preprocessed

python ../tools/prepare_amos_mri_task.py \
  --amos-root "$AMOS_ROOT" \
  --raw-data-base "$unetr_pp_raw_data_base" \
  --task-name Task007_AMOSMRI \
  --train-count 32 \
  --val-count 8

python ../unetr_pp/experiment_planning/nnFormer_plan_and_preprocess.py -t 7 --verify_dataset_integrity
