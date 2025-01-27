# code adapted, original source: https://github.com/caoyunkang/Segment-Any-Anomaly/blob/SAA-plus/demo.py

import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/winter/thesis-daniela/")

from pathlib import Path
from loguru import logger

from SegmentAnyAnomaly import *
from SegmentAnyAnomaly.utils.training_utils import *
from data_loader_teacher import DataLoader
from util.masks_to_bboxes import prepare_output, save_bboxes2csv


if __name__ == "__main__":
    import os

    gpu_id = 0

    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    weights_dir = "/data-fast/winter/weights"
    data_dir = "/data-fast/winter/isead/38k/birds/SegmentationMask"

    dino_config_file = "/home/winter/thesis-daniela/teacher/SegmentAnyAnomaly/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = f"{weights_dir}/groundingdino_swint_ogc.pth"
    sam_checkpoint = f"{weights_dir}/sam_vit_h_4b8939.pth"

    # threshold prompts and other settings
    box_threshold = 0.1  # confidence for detected bboxes
    text_threshold = 0.1  # confidence for text prompt inside of bbox
    k_mask = 20  # max no of anomalies
    defect_area_threshold = 0.05  # defect max area
    defect_min_area = 0  # defect min area
    eval_resolution = 1024
    # device = f"cpu"
    device = f"cuda:0"

    input_handler = DataLoader(Path(f"{data_dir}/GT/JPEGImages/"))

    # get the model
    model = SAA.Model(
        dino_config_file=dino_config_file,
        dino_checkpoint=dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        defect_min_area=defect_min_area,
        out_size=eval_resolution,
        device=device,
    )

    # anomaly and object prompt
    anomalies = [
        "bird."
    ]  # , "seagull.", "tern.", "albatross.", "fighter jet.", "kite.", "drone.", "airplane."]
    backgrounds = ["sea"]

    for background in backgrounds:
        for anomaly in anomalies:
            anomaly_path = (anomaly.removesuffix(".")).replace(" ", "_")
            logger.debug(anomaly_path)
            output_path = f"{data_dir}/teacher_soft_labels/output_{background}_{anomaly_path}_threshold_{k_mask}"
            os.makedirs(output_path, exist_ok=True)

            csv_path = Path(
                output_path,
                f"prediction.csv",
            )

            # delete csv if it exists already
            if csv_path.exists():
                csv_path.unlink()

            # open filestream to csv file
            with open(csv_path, mode="a", newline="") as csv_file:
                # create header for csv file
                header = pd.DataFrame(
                    columns=[
                        "image",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "obj_id",
                        "obj_class",
                        "confidence",
                    ]
                )

                # write header to csv file
                header.to_csv(csv_file, sep=";", header=True, index=False)

                for image, img_path in input_handler:
                    image_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_show = cv2.resize(
                        image_show, (eval_resolution, eval_resolution)
                    )

                    # property_text_prompts = f"the image of {background} have 1 similar {background}, with a maximum of 20 anomaly. The anomaly would not exceed 0.1 object area. "
                    model.set_property_text_prompts(
                        object_prompt=background,
                        object_number=1,
                        k_mask=k_mask,
                        defect_area_threshold=defect_area_threshold,
                        similar="similar",
                        verbose=True,
                    )

                    textual_prompts = [
                        [
                            anomaly,
                            background,
                        ]
                    ]  # detect prompts, filtered phrase

                    model.set_ensemble_text_prompts(textual_prompts, verbose=True)

                    model = model.to(device)
                    # get model prediction
                    score, appendix = model(image)

                    # similarity_map = appendix["similarity_map"]
                    # similarity_map = cv2.resize(
                    #    similarity_map, (eval_resolution, eval_resolution)
                    # )

                    score = cv2.resize(score, (eval_resolution, eval_resolution))
                    segmentations = prepare_output(score)

                    # save model output and csv output to file
                    save_bboxes2csv(csv_file, segmentations, img_path)

                    plt.imshow(score, alpha=1, cmap="gray")
                    plt.axis("off")  # no axes
                    plt.gca().set_frame_on(False)  # no frame

                    # Save the figure
                    plt.savefig(
                        os.path.join(
                            output_path,
                            f"{img_path.stem}.png",
                        ),
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close("all")
                    plt.clf()
                    logger.debug(
                        f"Predicted image {img_path.stem} with anomaly prompt <{anomaly}> and background prompt <{background}>"
                    )
