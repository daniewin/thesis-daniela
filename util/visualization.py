# code adapted, original source: ISEAD project (HITeC e.V.)

import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw

from util.data_classes import Segmentation


class Visualization:
    def __init__(self, visualization_dir: Path, image_dir: Path):
        self.visualization_dir = visualization_dir
        self.image_dir = image_dir
        self.tag = ""

    def create_and_save_visualization(
        self, preds: list[Segmentation], visualization_path, im_path
    ):
        # create pillow image
        pillow_img = Image.open(f"{im_path}").convert("RGB")
        w, h = pillow_img.size
        draw = ImageDraw.Draw(pillow_img)
        for pred in preds:
            self.set_tag(pred)
            self.create_visualization(w, h, draw, pred, pillow_img)
        # call save visualization function
        self._save_visualization(pillow_img, visualization_path)

    def draw_bbox(self, pred, draw: ImageDraw.ImageDraw, w, h):
        # draw bbox rectangle
        draw_xmin = float(max(pred.xmin * w - 3, 0))
        draw_ymin = float(max(pred.ymin * h - 3, 0))
        draw_xmax = float(min(pred.xmax * w + 3, w))
        draw_ymax = float(min(pred.ymax * h + 3, h))
        draw.rectangle(
            (draw_xmin, draw_ymin, draw_xmax, draw_ymax),
            fill=None,
            outline="yellow",
            width=3,
        )
        # draw tag
        text_pos = (draw_xmin, draw_ymax + 2)
        text_bbox = draw.textbbox(text_pos, self.tag)
        draw.rectangle(text_bbox, fill="yellow")
        draw.text(text_pos, self.tag, fill="white")

    def _save_visualization(self, pillow_img, visualization_path):
        # save image
        pillow_img.save(visualization_path, format="JPEG", quality=96)

    def set_tag(self, pred):
        self.tag = ""  # f"{pred.obj_class} {pred.confidence:.2f}"

    def create_visualization(self, w, h, draw, pred, pillow_img):
        self.draw_bbox(pred, draw, w, h)
        self._overlay_segmentation_mask(pillow_img, pred)

    def _overlay_segmentation_mask(self, image, segmentation):
        # Convert the segmentation mask to a PIL image
        mask = segmentation.mask.cpu().numpy()
        colored_mask = self._create_colored_mask(mask)

        # Resize the colored mask to the original image size
        colored_mask = colored_mask.resize(image.size, Image.Resampling.LANCZOS)

        # Extract the alpha channel from the colored mask
        alpha_mask = colored_mask.split()[3]

        # Overlay the mask on the image using the alpha channel as a mask
        image.paste(colored_mask, (0, 0), alpha_mask)

    def _create_colored_mask(self, mask):
        # Create an empty RGBA array with mask shape, initially fully transparent

        colored_array = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        # Check if the mask is 2D (grayscale)
        if mask.ndim == 2:
            # Set the RGB channels to green (0, 255, 0) where the mask is True
            colored_array[mask, :3] = [0, 255, 0]

            # Set the alpha channel to semi-transparent where the mask is True, else transparent
            colored_array[mask, 3] = 100  # alpha 0 = transparent, alpha 255 = opaque

        # Check if the mask is 3D (RGB, multiclass segmentation)
        elif mask.ndim == 3 and mask.shape[2] == 3:
            # Use the existing colors from the mask
            colored_array[:, :, :3] = mask

            # Identify background areas
            background = np.all(mask == [0, 0, 0], axis=-1)
            # Set the alpha channel to semi-transparent for segmented areas, else transparent
            colored_array[~background, 3] = (
                100  # alpha 0 = transparent, alpha 255 = opaque
            )

        else:
            raise ValueError(
                "Invalid mask format. Mask must be either 2D (grayscale) or 3D (RGB)."
            )

        # Convert the array to a PIL Image
        colored_mask = Image.fromarray(colored_array).convert("RGBA")
        return colored_mask
