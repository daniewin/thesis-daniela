# code adapted, original source: ISEAD project (HITeC e.V.)

import pandas as pd
import torch

from scipy.ndimage import label, binary_dilation
from pathlib import Path

from util.data_classes import Segmentation, BBox, ObjectClass


def save_bboxes2csv(
    csv_file,
    prediction: list[BBox] | list[Segmentation],
    img_path: Path,
) -> None:
    for pred in prediction:
        df = pd.DataFrame(
            {
                "image": img_path.name,
                **pred.get_bbox_dict(),
            },
            index=[0],
        )
        # save csv file
        df.to_csv(csv_file, sep=";", header=False, index=False)


def prepare_output(output) -> list[Segmentation]:
    # find connected components
    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    output = binary_dilation(input=output, structure=structure, iterations=3)
    labeled_array, num_features = label(input=output, structure=structure)
    masks = []
    for idx in range(1, num_features + 1):
        mask = labeled_array == idx
        masks.append(mask)
    # convert to list of Segmentation
    segmentations = []
    for idx, mask in enumerate(masks):
        confidence = output[mask].mean()
        mask = torch.from_numpy(mask)
        mask = mask.squeeze(0)
        seg = Segmentation(
            mask=mask,
            confidence=confidence,
            obj_id=idx,
            obj_class=ObjectClass.UNKNOWN,
        )
        segmentations.append(seg)

    return segmentations
