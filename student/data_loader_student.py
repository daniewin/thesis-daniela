# code adapted, original source: ISEAD project (HITeC e.V.)

import torch
import numpy as np
import sys

sys.path.append("/home/winter/thesis-daniela/")

from dataclasses import dataclass
from loguru import logger
from typing import Iterator
from PIL import Image
from pathlib import Path


@dataclass
class InputHandler:

    def __init__(
        self, config: dict, transform: tuple, data_dir: Path, split_file: Path, mode
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.image_dir = Path(self.data_dir) / Path(
            config["data"]["input"]["image_dir"]
        )
        self.label_dir = Path(self.data_dir) / Path(
            config["data"]["input"][str(self.mode + "_label_dir")]
        )
        self.soft_label_filename_ending = config["data"]["input"][
            "soft_label_filename_ending"
        ]

        self.transform_image = transform[0]
        self.transform_label = transform[1]
        self.split_file = split_file
        self.image_paths = self._find_images()
        self.temperature = config["training"]["temperature"]
        self.batch_size = config["training"]["batch_size"]

    # get test data filenames
    def _read_txt(self, file_path):
        test = []
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                test.append(line.removesuffix("\n"))
        return test

    def _find_images(self) -> list[Path]:
        logger.debug(f"Finding images in {self.image_dir}")

        image_paths: list[Path] = []

        if self.split_file:
            logger.debug(f"with split file {self.split_file}.")
            # get data filenames
            split = self._read_txt(self.split_file)

        # find all images in the input path with the supported file types
        for file_type in ["JPEG", "jpeg", "JPG", "jpg"]:
            images_list = []
            images_list.extend(self.image_dir.glob(f"*.{file_type}"))

            for image in images_list:
                # positive dataset
                if str(image.stem).removesuffix(f".{file_type}") in split:
                    # negative dataset
                    # if (str(image.stem) + f".{file_type}") in split:
                    image_paths.append(image)
        logger.debug(f"Found {len(image_paths)} images.")

        return image_paths

    def _load_image(self, image_path: Path) -> torch.Tensor:
        # load image
        image = Image.open(image_path)
        # apply transforms
        image = self.transform_image(image)
        return image

    def _load_label(self, image_path: Path) -> torch.Tensor:
        label_path = Path(
            f"{self.label_dir}/{image_path.stem}{self.soft_label_filename_ending}.png"
        )
        # load image
        label = Image.open(label_path)
        label = label.convert("L")
        # if GT used
        # label = np.array(label, dtype=np.float32)
        # label = np.where(label == 64, 255.0, 0.0)

        # apply transforms
        label = self.transform_label(label)
        label /= self.temperature
        return label

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:

        batch = []
        labels = []
        filenames = []

        for idx in range(len(self.image_paths)):
            im_path = self.image_paths[idx]
            image = self._load_image(im_path)
            label = self._load_label(im_path)
            filename = Path(im_path.stem)

            # make sure that image is in square shape
            if image.size()[2] == image.size()[1]:
                batch.append(image)
                labels.append(label)
                filenames.append(filename)

            # in case batch is already full
            if len(batch) == self.batch_size:
                yield torch.stack(batch), torch.stack(labels), filenames
                batch = []
                labels = []
                filenames = []

        # for last batch that might be smaller
        if len(batch) > 1:
            yield torch.stack(batch), torch.stack(labels), filenames
