import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import sys

sys.path.append("/home/winter/thesis-daniela/")

from loguru import logger


def read_soft_labels(image_prefix, teacher_dirs):
    masks = []
    for teacher_dir in teacher_dirs:
        mask_path = glob.glob(f"{teacher_dir}/{image_prefix}*.*")[0]
        mask = Image.open(mask_path).convert("L")
        masks.append(np.array(mask))
    return masks


def majority_voting(masks, threshold=6):
    masks = np.array(masks)
    mask_shape = masks[0].shape
    combined_mask = np.zeros(mask_shape, dtype=np.float32)
    non_black = np.sum(masks > 0, axis=0)
    # voting
    voted_pixels = non_black >= threshold
    # for all pixels that are voted for, calculate average
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            if voted_pixels[i, j]:
                non_black_values = masks[:, i, j][masks[:, i, j] > 0]
                combined_mask[i, j] = np.mean(non_black_values)
    return combined_mask


def save_combined_label(output_path, combined_label):
    combined_label_im = Image.fromarray(np.uint8(combined_label))
    combined_label_im.save(output_path)


def combine_labels(image_dir, chosen_teachers, output_dir, threshold, i, strategy):
    os.makedirs(output_dir, exist_ok=True)
    image_file_paths = glob.glob(f"{image_dir}*.*", recursive=True)
    image_file_paths = [
        path
        for path in image_file_paths
        if path.lower().endswith((".png", ".jpeg", ".jpg"))
    ]

    counter = 0
    for im_path in image_file_paths:
        logger.info(f"processing image {counter}")
        image_prefix = Path(im_path).stem

        masks = read_soft_labels(image_prefix, chosen_teachers)

        if strategy == "mean":
            mean_masks_array = np.mean(np.stack(masks, axis=0), axis=0)
            combined_label = Image.fromarray(np.uint8(mean_masks_array))

        if strategy == "majority":
            combined_label = majority_voting(masks, threshold=threshold)

        output_path = os.path.join(output_dir, f"{image_prefix}.png")
        save_combined_label(output_path, combined_label)
        counter += 1


data_dir = "/data-fast/winter/isead/38k/birds/SegmentationMask"
label_dir = f"{data_dir}/teacher_soft_labels"
image_dir = f"{data_dir}/GT_birds/JPEGImages/"
threshold = 2
strategy = "mean"

chosen_teachers_list = [
    ["shearwater", "bird", "marine bird", "oceanic bird", "car"],
    ["pelican", "shearwater", "cat", "tern", "dolphin"],
    ["gull", "bird", "tern", "penguin", "oceanic bird"],
]


for i, chosen_teachers in enumerate(chosen_teachers_list):
    chosen_teachers = [
        f"{label_dir}/output_{teacher}_threshold_20" for teacher in chosen_teachers
    ]

    output_dir = f"{label_dir}/majority_group_{i}_threshold_{threshold}"
    combine_labels(image_dir, chosen_teachers, output_dir, threshold, i, strategy)
