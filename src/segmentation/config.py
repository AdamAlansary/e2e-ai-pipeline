import torch
from torchvision.transforms import v2


# Define image and mask paths
BASE_PATH = "data/human_segmentation_kaggle"
IMAGE_DATASET_PATH = "data/human_segmentation_kaggle/images"
MASK_DATASET_PATH = "data/human_segmentation_kaggle/masks"
CSV_PATH = "data/human_segmentation_kaggle/df.csv"
TRAIN_DATASET_CSV = "data/human_segmentation_kaggle/train.csv"
VAL_DATASET_CSV = "data/human_segmentation_kaggle/val.csv"
TEST_DATASET_CSV = "data/human_segmentation_kaggle/test.csv"

# Define the random seed for splitting the data
SEED = 42

# Define the device to be used for training and evaluation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# Define the number of channels in the input, number of classes, and number of levels in the model
NUM_CHANNELS = 3
NUM_CLASSES = 1

# initialize learning rate, number of epochs to train for, the batch size, and the number of workers
INIT_LR = 0.0001
NUM_EPOCHS = 15
BATCH_SIZE = 3
NUM_WORKERS = 8

# Define the input image dimensions
INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 640

# Define the mean and standard deviation values and their inverse for normalizing and denormalizing the images
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
INV_MEAN, INV_STD = [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]

# Define the image and mask transformations needed for the model
IMAGE_TRANSFORMS = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)),
    v2.Normalize(mean=MEAN, std=STD),
])
MASK_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
])

# Define threshold to filter weak predictions
THRESHOLD = 0.4
