#!/bin/sh

DATASET_PATH=/project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR/DATASET_Synapse

export PYTHONPATH=/project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR/unetr_plus_plus-main
export RESULTS_FOLDER=/project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR/unetr_plus_plus-main/unetr_pp/results/unetr_pp_synapse
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python /project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR/unetr_plus_plus-main/unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2 0
