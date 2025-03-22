import config.config as cfg
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def deeplabv3():
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier = DeepLabHead(2048, cfg.NUM_CLASSES)
    return model
