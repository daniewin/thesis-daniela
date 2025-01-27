# code adapted, original source: ISEAD project (HITeC e.V.)

import torch
import cv2

from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from typing import Iterator


@dataclass
class DataLoader:
    image_input_path: Path
    image_paths: list[Path] = field(init=False)

    def __post_init__(self):
        self.image_paths = self._find_images(self.image_input_path)

    def _find_images(self, image_input_path: Path) -> list[Path]:
        """
        Find all image files in the specified input path.
        """
        logger.debug(f"Finding images in {image_input_path}")

        image_paths: list[Path] = []
        # Find all images in the input path with the supported file types
        for file_type in ["JPEG", "jpeg", "JPG", "jpg", "PNG", "png"]:
            image_paths.extend(image_input_path.glob(f"*.{file_type}"))
        logger.debug(f"Found {len(image_paths)} images.")

        return image_paths

    def _load_image(self, im_path: Path) -> torch.Tensor:
        # load image
        image = cv2.imread(str(im_path), cv2.IMREAD_COLOR)

        return image

    def __iter__(self) -> Iterator[tuple[torch.Tensor, Path]]:
        for im_path in self.image_paths:
            yield self._load_image(im_path), im_path
