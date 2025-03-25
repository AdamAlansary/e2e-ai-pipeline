# https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data
import cv2
import pandas as pd
import config.config as cfg
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
	def __init__(self, csv_path):
		# Store the image and mask filepaths, and augmentation transforms
		super().__init__()

		df = pd.read_csv(csv_path)
		df = pd.DataFrame(df)

		# Extract image and mask paths from respective DataFrame columns
		self.image_paths = df['images'].values#[:(len(df['images'].values) // 10)]
		self.mask_paths = df['masks'].values#[:(len(df['masks'].values) // 10)]

		self.image_transforms = cfg.IMAGE_TRANSFORMS
		self.mask_transforms = cfg.MASK_TRANSFORMS
		
		
	def __len__(self):
		return len(self.image_paths)
	

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		mask_path = self.mask_paths[idx]
		
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Swap image channels from BGR to RGB
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale mode
		
		# Apply the transformations to the image and mask
		image = self.image_transforms(image)
		mask = self.mask_transforms(mask)
		
		return (image, mask)
