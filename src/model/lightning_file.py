import torch
import config.config as cfg
import lightning.pytorch as pl
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from src.utils.dataset import SegmentationDataset


class HumanSegmentation(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self.forward(inputs)

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(outputs['out'], masks)

        self.log("training_loss", loss)
        print("   training batch " + str(batch_idx) + " loss: " + str(loss.item()))

        return loss


    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self.forward(inputs)

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(outputs['out'], masks)

        self.log("validation_loss", loss)
        print("   validation batch " + str(batch_idx) + " loss: " + str(loss.item()))


    def test_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self.forward(inputs)
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(outputs['out'], masks)

        # Apply sigmoid and threshold to obtain binary predictions.
        probs = torch.sigmoid(outputs['out'])
        preds = (probs > 0.1).float()

        # Compute Dice score for binary segmentation.
        smooth = 1e-6
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth)
        dice_score = dice.mean()
        dice_loss = 1 - dice_score

        # Ensure predictions and masks are of shape (B, H, W)
        # If they have a channel dimension of size 1, squeeze it.
        if preds.shape[1] == 1:
            preds_bin = preds.squeeze(1)
        else:
            preds_bin = preds
        if masks.shape[1] == 1:
            masks_bin = masks.squeeze(1)
        else:
            masks_bin = masks

        # Pixel Accuracy: fraction of correctly predicted pixels.
        pixel_accuracy = (preds_bin == masks_bin).float().mean()

        # IoU: For each image compute intersection over union, then average.
        intersection_iou = (preds_bin * masks_bin).sum(dim=(1, 2))
        union_iou = ((preds_bin + masks_bin) >= 1).float().sum(dim=(1, 2))
        iou = (intersection_iou + smooth) / (union_iou + smooth)
        mean_iou = iou.mean()

        # Log metrics. Using on_epoch=True so that the average over the test epoch is reported.
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_dice_loss", dice_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pixel_accuracy", pixel_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mean_iou", mean_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "test_loss": loss,
            "test_dice_loss": dice_loss,
            "test_pixel_accuracy": pixel_accuracy,
            "test_mean_iou": mean_iou
        }
        

    ### SETUP ###

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.INIT_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6, cooldown=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}


    def train_dataloader(self):
        train_data = SegmentationDataset(csv_path=cfg.TRAIN_DATASET_CSV)
        train_dl = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        return train_dl


    def val_dataloader(self):
        val_data = SegmentationDataset(csv_path=cfg.VAL_DATASET_CSV)
        val_dl = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        return val_dl


    def test_dataloader(self):
        test_data = SegmentationDataset(csv_path=cfg.TEST_DATASET_CSV)
        test_dl = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, persistent_workers=True)
        return test_dl
