import torch
import numpy as np
import json
import torch.optim as optim
import os
import segmentation_models_pytorch as smp
import pandas as pd
import sys
import wandb

sys.path.append("/home/winter/thesis-daniela/")

from torchvision import transforms
from loguru import logger
from pathlib import Path

from data_loader_student import InputHandler
from util.metrics import bbox_metrics
from util.masks_to_bboxes import prepare_output, save_bboxes2csv


class Trainer:
    def __init__(self, config, transform, model, device):

        server = config["server"]
        mode = config["mode"]
        data_dir = Path(config["data"][str("data_dir_" + server)])

        self.max_bbox_f1 = 0
        self.split_train_path = Path(data_dir) / Path(
            config["data"]["input"]["split_train_path"]
        )
        self.split_val_path = Path(data_dir) / Path(
            config["data"]["input"]["split_val_path"]
        )
        self.csv_path_gt = Path(data_dir) / Path(config["data"]["input"]["csv_path_gt"])

        self.test_images_out_dir = Path(data_dir) / Path(
            config["data"]["output"]["test_images_out_dir"]
        )

        self.train_loader = InputHandler(
            config, transform, data_dir, self.split_train_path, mode
        )
        self.val_loader = InputHandler(
            config, transform, data_dir, self.split_val_path, mode
        )

        self.model = model
        self.checkpoint_dir = Path(config["model"][str("checkpoint_dir_" + server)])
        self.checkpoint_file = Path(config["model"]["checkpoint_file"])
        self.device = device

        # define loss function and optimizer
        lr = config["training"]["lr"]

        if config["training"]["optimizer"] == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        if config["training"]["loss"] == "Dice":
            self.criterion = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE, from_logits=False, eps=1e-07
            )

        self.num_epochs = config["training"]["num_epochs"]
        self.output_threshold = 1  # config["data"]["output"]["binary_threshold"]

    def train(self):

        step = 0
        for epoch in range(self.num_epochs):
            # validation step
            self.validate(step, epoch)

            self.model.train()

            for images, masks, _ in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logger.info(f"this is batch {step} in epoch {epoch}")
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                normalized_masks = torch.sigmoid(outputs)

                with torch.no_grad():
                    train_loss_num = loss.item() / len(images)

                    outputs_binary = (normalized_masks > self.output_threshold).to(
                        torch.uint8
                    )

                # wandb.log({"train_loss": train_loss_num}, step=step)

                step += 1

            # wandb.log(
            #    {"Train GT Image": [wandb.Image(masks, caption="GT Image")]}, step=step
            # )
            wandb.log(
                {"Train Output Image": [wandb.Image(outputs, caption="Output Image")]},
                step=step,
            )

    def validate(self, step, epoch):
        self.model.eval()
        val_predictions = []

        with torch.no_grad():
            for images, masks, filenames in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                val_predictions.append((outputs.cpu().numpy(), filenames))

        # save predictions to csv
        os.makedirs(self.test_images_out_dir, exist_ok=True)
        csv_path_val_pred = Path(
            f"{self.test_images_out_dir}/validation_prediction.csv"
        )

        # delete csv if it exists already
        if csv_path_val_pred.exists():
            csv_path_val_pred.unlink()

        # open filestream to csv file
        with open(csv_path_val_pred, mode="a", newline="") as csv_file:
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

            for pred_tuple in val_predictions:
                pred_batch, filenames = pred_tuple

                for j, pred in enumerate(pred_batch):
                    filename = filenames[j]
                    pred = pred[0] * 255
                    pred = np.where(pred > self.output_threshold, 255, 0)
                    segmentations = prepare_output(pred)
                    # save output to csv file
                    save_bboxes2csv(csv_file, segmentations, filename)

            bbox_precision, bbox_recall, bbox_f1 = bbox_metrics(
                csv_path_val_pred, self.csv_path_gt, self.split_val_path
            )

            # wandb.log(
            #    {"Val GT Image": [wandb.Image(masks, caption="GT Image")]}, step=step
            # )
            wandb.log(
                {"Val Output Image": [wandb.Image(outputs, caption="Output Image")]},
                step=step,
            )

            wandb.log({"epoch": epoch, "bbox precision": bbox_precision}, step=step)
            wandb.log({"epoch": epoch, "bbox recall": bbox_recall}, step=step)
            wandb.log({"epoch": epoch, "bbox f1": bbox_f1}, step=step)

            # save model checkpoint
            if bbox_f1 >= self.max_bbox_f1:
                self.max_bbox_f1 = bbox_f1
                checkpoint_path = Path(
                    self.checkpoint_dir, f"{self.checkpoint_file}_{epoch}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                # wandb.save(checkpoint_path)


def main():

    # load config from JSON file
    with open("config.json") as f:
        config = json.load(f)

    # transformation
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

    # initialize w&b
    run = wandb.init(
        project="thesis-daniela",
        entity="dani-win",
        name=f"GT_birds_student_bird_sea_20_0.05_GT",
    )

    # create logfile
    logger.add(config["data"]["output"]["logfile_path"])

    # load model
    model = smp.PAN(
        encoder_name="resnet34", encoder_weights="imagenet", activation="sigmoid"
    )
    # move model to device
    device = torch.device("cuda")
    model.to(device)
    logger.debug(model)

    # training
    trainer = Trainer(config, transform, model, device)
    logger.info(config)
    trainer.train()
    # run.finish()


if __name__ == "__main__":
    main()
