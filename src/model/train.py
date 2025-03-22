import config.config as cfg
import lightning.pytorch as pl
from model_deeplabv3 import deeplabv3
from lightning_file import HumanSegmentation
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint


if __name__ == '__main__':
    # Run this in terminal first: mlflow server --host 127.0.0.1 --port 8080
    mlf_logger = MLFlowLogger(experiment_name="Human Segmentation Test", run_name="Test run 1 (266 images)", log_model=False, tracking_uri="http://127.0.0.1:8080")
    # mlf_logger = MLFlowLogger(experiment_name="Binary Segmentation Finetuning", run_id="320bc56838e3412ca8dbc340b2a6c8f9", log_model=False, tracking_uri="http://127.0.0.1:8080")

    checkpoint_callback = ModelCheckpoint(dirpath=f"mlruns/{mlf_logger.experiment_id}/{mlf_logger.run_id}/artifacts",
                                          filename="human-seg-{epoch}-{validation_loss:.5f}",
                                          monitor="validation_loss",
                                          mode="min",
                                          verbose=True,
                                          save_top_k=3)
    
    model = deeplabv3()
    model_lightning = HumanSegmentation(model=model)

    trainer = pl.Trainer(max_epochs=cfg.NUM_EPOCHS,
                        logger=mlf_logger,
                        callbacks=[checkpoint_callback],
                        accelerator="gpu",
                        # precision=16,
                        devices=-1)

    trainer.fit(model=model_lightning)
    # trainer.fit(model=model, ckpt_path="mlruns/163865829670612271/320bc56838e3412ca8dbc340b2a6c8f9/artifacts/pid-binary-segmentation-epoch=9-validation_loss=0.01625.ckpt")
