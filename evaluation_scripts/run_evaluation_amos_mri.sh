#!/bin/sh

PROJ=/project2/ruishanl_1185/zrui6736/Ruishan_Multiorgan_UNETR_new
DATASET_PATH=$PROJ/DATASET_AMOSMRI

export PYTHONPATH=$PROJ
export RESULTS_FOLDER=$PROJ/unetr_pp/results/output_amos_mri
export unetr_pp_raw_data_base=$DATASET_PATH/unetr_pp_raw
export unetr_pp_preprocessed=$DATASET_PATH/unetr_pp_preprocessed
export nnFormer_def_n_proc=1

VAL_DIR=$RESULTS_FOLDER/unetr_pp/3d_fullres/Task007_AMOSMRI/unetr_pp_trainer_amos_mri__unetr_pp_Plansv2.1/fold_0/validation_raw
mkdir -p "$VAL_DIR"
rm -f "$VAL_DIR"/*.nii.gz "$VAL_DIR"/*.npz "$VAL_DIR"/*.npy

python $PROJ/unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_amos_mri 7 0 -val --disable_postprocessing_on_folds
