# code adapted, original source: ISEAD project (HITeC e.V.)

import os
import json
import numpy as np
import pandas as pd
import sys

sys.path.append("/home/winter/thesis-daniela/")

from PIL import Image
from loguru import logger
from pathlib import Path

from masks_to_bboxes import prepare_output, save_bboxes2csv
from visualization import Visualization


class CSV_Creator:
    def __init__(self, config, visualization):

        data_dir = Path(config["data"][str("data_dir_" + config["server"])])
        self.test_images_in_dir = Path(data_dir) / Path(
            config["data"]["input"]["image_dir"]
        )
        self.binary_threshold = config["data"]["output"]["binary_threshold"]
        self.test_images_out_dir = Path(data_dir) / Path(
            config["data"]["output"]["test_images_out_dir"]
        )
        self.csv_path = Path(f"{self.test_images_out_dir}/prediction.csv")
        self.visualization = visualization

    def create_csv(self):

        predictions = []

        for filename in os.listdir(self.test_images_out_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                # Construct full file path
                file_path = os.path.join(self.test_images_out_dir, filename)

                # Open the image file
                with Image.open(file_path) as img:
                    # Convert the image to a numpy array
                    img = img.convert("L")
                    img_array = np.array(img)
                    predictions.append((img_array, Path(filename)))

        # delete csv if it exists already
        if self.csv_path.exists():
            self.csv_path.unlink()

        visualization_dir = f"{self.test_images_out_dir}"
        im_dir = f"{self.test_images_in_dir}"
        visualizer = Visualization(visualization_dir, im_dir)

        # open filestream to csv file
        with open(self.csv_path, mode="a", newline="") as csv_file:
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

            for pred_tuple in predictions:
                pred, filename = pred_tuple
                pred = pred * 255
                pred = np.where(pred > self.binary_threshold, 255, 0)

                segmentations = prepare_output(pred)
                # save model output and csv output to file
                save_bboxes2csv(csv_file, segmentations, filename)

                # save bbox visualization and binary segmentation mask
                if self.visualization:
                    # mask_image = Image.fromarray(pred.astype(np.uint8))
                    # save the mask as a PNG file
                    # mask_image.save(f'{self.test_images_out_dir}/prediction_{filename}.png')

                    visualization_path = (
                        f"{self.test_images_out_dir}/overlay_{filename}.png"
                    )
                    im_path = f"{self.test_images_in_dir}/{filename}.jpg"
                    visualizer.create_and_save_visualization(
                        segmentations, visualization_path, im_path
                    )


def main():

    # load config from JSON file
    with open("config.json") as f:
        config = json.load(f)

    # create logfile
    logger.add(config["data"]["output"]["logfile_path"])

    # bbox visualization
    visualization = False
    csv_creator = CSV_Creator(config, visualization)

    logger.info(config)
    # create a csv file for all detections in soft labels in the test output path
    csv_creator.create_csv()


if __name__ == "__main__":
    main()
