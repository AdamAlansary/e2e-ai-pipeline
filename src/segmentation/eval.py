import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from src.segmentation.model_deeplabv3 import deeplabv3
from src.segmentation.lightning_file import HumanSegmentation


if __name__ == '__main__':
    # Run this in terminal first: mlflow server --host 127.0.0.1 --port 8080
    mlf_logger = MLFlowLogger(experiment_name="Human Segmentation Finetuning", run_id="640abc12e9b2419f910196ffead7e2f8", log_model=False, tracking_uri="http://127.0.0.1:8080")

    model = deeplabv3()
    model_lightning = HumanSegmentation(model=model)

    trainer = pl.Trainer(logger=mlf_logger,
                        accelerator="gpu",
                        devices=-1)

    trainer.test(model=model_lightning, ckpt_path="mlruns/163865829670612271/640abc12e9b2419f910196ffead7e2f8/artifacts/pid-binary-segmentation-epoch=9-validation_loss=0.01629.ckpt")
