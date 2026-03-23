#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path


MRI_ID_THRESHOLD = 500


def parse_amos_id(filename: str):
    m = re.match(r"^amos_(\d+)\.nii\.gz$", filename)
    if m is None:
        return None
    return int(m.group(1))


def list_mri_cases(folder: Path):
    cases = []
    for p in sorted(folder.glob("amos_*.nii.gz")):
        amos_id = parse_amos_id(p.name)
        if amos_id is None:
            continue
        if amos_id >= MRI_ID_THRESHOLD:
            cases.append((amos_id, p))
    return cases


def ensure_clean_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob("*.nii.gz"):
        p.unlink()


def copy_train_case(src_img: Path, src_lbl: Path, dst_images_tr: Path, dst_labels_tr: Path):
    case_name = src_img.name[:-7]  # strip .nii.gz
    shutil.copy2(src_img, dst_images_tr / f"{case_name}_0000.nii.gz")
    shutil.copy2(src_lbl, dst_labels_tr / f"{case_name}.nii.gz")
    return case_name


def copy_test_case(src_img: Path, dst_images_ts: Path):
    case_name = src_img.name[:-7]  # strip .nii.gz
    shutil.copy2(src_img, dst_images_ts / f"{case_name}_0000.nii.gz")
    return case_name


def build_dataset_json(training_case_names, test_case_names):
    return {
        "name": "AMOSMRI",
        "description": "AMOS MRI subset (id >= 500) for UNETR++ Task007",
        "tensorImageSize": "3D",
        "reference": "https://jiyuanfeng.github.io/AMOS/",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "modality": {"0": "MRI"},
        "labels": {
            "0": "background",
            "1": "spleen",
            "2": "right kidney",
            "3": "left kidney",
            "4": "gall bladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "aorta",
            "9": "postcava",
            "10": "pancreas",
            "11": "right adrenal gland",
            "12": "left adrenal gland",
            "13": "duodenum",
            "14": "bladder",
            "15": "prostate/uterus",
        },
        "numTraining": len(training_case_names),
        "numTest": len(test_case_names),
        "training": [
            {
                "image": f"./imagesTr/{c}.nii.gz",
                "label": f"./labelsTr/{c}.nii.gz",
            }
            for c in training_case_names
        ],
        "test": [f"./imagesTs/{c}.nii.gz" for c in test_case_names],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Task007_AMOSMRI from amos22 folder")
    parser.add_argument(
        "--amos-root",
        type=Path,
        default=Path("/Users/weisu/Ruishan_Multiorgan/UNETR++/amos22"),
        help="Path to AMOS root with imagesTr/labelsTr/imagesTs",
    )
    parser.add_argument(
        "--raw-data-base",
        type=Path,
        default=Path("/Users/weisu/Ruishan_Multiorgan/UNETR++/unetr_plus_plus-main/DATASET_AMOSMRI/unetr_pp_raw"),
        help="Path used as unetr_pp_raw_data_base",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="Task007_AMOSMRI",
        help="Task folder name under unetr_pp_raw_data",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=32,
        help="Number of first MRI cases for training split",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=8,
        help="Number of following MRI cases for validation split",
    )
    args = parser.parse_args()

    images_tr_src = args.amos_root / "imagesTr"
    labels_tr_src = args.amos_root / "labelsTr"
    images_ts_src = args.amos_root / "imagesTs"

    if not images_tr_src.is_dir() or not labels_tr_src.is_dir() or not images_ts_src.is_dir():
        raise RuntimeError(f"AMOS folder is incomplete: {args.amos_root}")

    all_mri_train = list_mri_cases(images_tr_src)
    needed = args.train_count + args.val_count
    if len(all_mri_train) < needed:
        raise RuntimeError(
            f"Not enough MRI training cases in imagesTr (>=500). "
            f"Need at least {needed}, got {len(all_mri_train)}."
        )

    # First 32 and last 8 (from the first 40 MRI cases), per user requirement.
    selected_40 = all_mri_train[:needed]
    selected_train = selected_40[: args.train_count]
    selected_val = selected_40[args.train_count : args.train_count + args.val_count]

    # We keep all 40 in training set files (labels available), and split 32/8 in trainer.
    selected_training_cases = selected_train + selected_val

    all_mri_test = list_mri_cases(images_ts_src)

    task_root = args.raw_data_base / "unetr_pp_raw_data" / args.task_name
    images_tr_dst = task_root / "imagesTr"
    labels_tr_dst = task_root / "labelsTr"
    images_ts_dst = task_root / "imagesTs"

    ensure_clean_dir(images_tr_dst)
    ensure_clean_dir(labels_tr_dst)
    ensure_clean_dir(images_ts_dst)

    copied_training_case_names = []
    for _, src_img in selected_training_cases:
        src_lbl = labels_tr_src / src_img.name
        if not src_lbl.is_file():
            raise RuntimeError(f"Missing label for {src_img.name}")
        copied_training_case_names.append(
            copy_train_case(src_img, src_lbl, images_tr_dst, labels_tr_dst)
        )

    copied_test_case_names = []
    for _, src_img in all_mri_test:
        copied_test_case_names.append(copy_test_case(src_img, images_ts_dst))

    dataset = build_dataset_json(copied_training_case_names, copied_test_case_names)
    with open(task_root / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Task created: {task_root}")
    print(f"Training cases copied: {len(copied_training_case_names)} (32 train + 8 val split in trainer)")
    print(f"Test MRI cases copied: {len(copied_test_case_names)}")
    print("Validation split (fixed in trainer):")
    print("  train:", [c[1].name[:-7] for c in selected_train])
    print("  val  :", [c[1].name[:-7] for c in selected_val])


if __name__ == "__main__":
    main()
