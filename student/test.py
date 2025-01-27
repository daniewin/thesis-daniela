import torch
import json
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import sys

sys.path.append("/home/winter/thesis-daniela/")

from torchvision import transforms
from PIL import Image
from loguru import logger
from pathlib import Path

from data_loader_student import InputHandler
from util.masks_to_bboxes import prepare_output, save_bboxes2csv
from util.visualization import Visualization


class Tester:
    def __init__(
        self,
        config,
        transform,
        model,
        device,
        visualization,
        anomaly,
        maj,
        group,
        threshold_no,
        threshold_area,
    ):

        mode = config["mode"]
        data_dir = Path(config["data"][str("data_dir_" + config["server"])])
        split_test_path = Path(data_dir) / Path(
            config["data"]["input"]["split_test_path"]
        )
        self.test_loader = InputHandler(
            config, transform, data_dir, split_test_path, mode
        )
        self.test_images_in_dir = Path(data_dir) / Path(
            config["data"]["input"]["image_dir"]
        )
        self.test_images_out_dir = Path(data_dir) / Path(
            config["data"]["output"]["test_images_out_dir"]
        )

        self.device = device
        self.model = model
        self.output_threshold = config["data"]["output"]["binary_threshold"]
        self.visualization = visualization

    def test(self):

        predictions = []
        counter = 0
        for inputs, _, filenames in self.test_loader:

            logger.info(f"Test predictions for batch number {counter}.")

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)

            predictions.append((outputs.cpu().numpy(), filenames))
            counter += 1

        # save predictions
        logger.info(len(predictions))
        csv_path = Path(f"{self.test_images_out_dir}/test_prediction.csv")

        # delete csv if it exists already
        if csv_path.exists():
            csv_path.unlink()

        if self.visualization:
            visualization_dir = f"{self.test_images_out_dir}"
            im_dir = f"{self.test_images_in_dir}"
            visualizer = Visualization(visualization_dir, im_dir)

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

            for pred_tuple in predictions:
                pred_batch, filenames = pred_tuple

                for j, pred in enumerate(pred_batch):
                    filename = filenames[j]
                    pred = pred[0] * 255
                    pred = np.where(pred > self.output_threshold, 255, 0)
                    segmentations = prepare_output(pred)
                    # save model output and csv output to file
                    save_bboxes2csv(csv_file, segmentations, filename)

                    if self.visualization:

                        mask_image = Image.fromarray(pred.astype(np.uint8))
                        # save the mask as a PNG file
                        mask_image.save(
                            f"{self.test_images_out_dir}/prediction_{filename}.png"
                        )
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

    # define transformation
    transform_image = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([256, 256])]
    )
    transform_label = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
        ]
    )

    transform = (transform_image, transform_label)

    anomaly = ""
    group = ""
    maj = ""
    threshold_no = 0
    threshold_area = 0

    # checkpoint
    checkpoint_dir = config["model"][str("checkpoint_dir_" + config["server"])]
    checkpoint_file = str(config["model"]["checkpoint_file"] + ".pth")

    checkpoint_path = Path(checkpoint_dir, checkpoint_file)

    # load model
    model = smp.PAN(
        encoder_name="resnet34", encoder_weights="imagenet", activation="sigmoid"
    )

    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    model.eval()

    # move model to device
    device = torch.device("cuda")
    model.to(device)

    logger.debug(model)
    visualization = False
    tester = Tester(
        config,
        transform,
        model,
        device,
        visualization,
        anomaly,
        maj,
        group,
        threshold_no,
        threshold_area,
    )
    logger.info(config)
    tester.test()


if __name__ == "__main__":
    main()
