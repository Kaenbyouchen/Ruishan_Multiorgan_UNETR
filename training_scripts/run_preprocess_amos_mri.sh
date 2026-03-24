#!/bin/sh

PROJ=/project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR_new
DATASET_PATH=$PROJ/DATASET_AMOSMRI

export PYTHONPATH=$PROJ/unetr_plus_plus-main
export RESULTS_FOLDER=$PROJ/unetr_plus_plus-main/unetr_pp/results/output_amos_mri
export unetr_pp_raw_data_base=$DATASET_PATH/unetr_pp_raw
export unetr_pp_preprocessed=$DATASET_PATH/unetr_pp_preprocessed

python $PROJ/unetr_plus_plus-main/unetr_pp/experiment_planning/nnFormer_plan_and_preprocess.py -t 7 --verify_dataset_integrity