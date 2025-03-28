# https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data
import cv2
import random
import pandas as pd
import src.segmentation.config as cfg
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
	def __init__(self, csv_path):
		super().__init__()

		df = pd.read_csv(csv_path)
		df = pd.DataFrame(df)

		# Extract image and mask paths from respective DataFrame columns
		self.image_paths = df['images'].values#[:(len(df['images'].values) // 10)]
		self.mask_paths = df['masks'].values#[:(len(df['masks'].values) // 10)]

		self.image_transforms = cfg.IMAGE_TRANSFORMS
		self.mask_transforms = cfg.MASK_TRANSFORMS
		
		
	def __len__(self):
		# Return twice the number of samples: one for the original and one for the augmented image
		return 2 * len(self.image_paths)
	

	def __getitem__(self, idx):
		# Determine whether this sample should be original or augmented.
		# If idx is less than the number of images, return original.
		# Otherwise, return the augmented version of the sample at index (idx - len(image_paths)).
		is_augmented = idx >= len(self.image_paths)
		real_idx = idx if not is_augmented else idx - len(self.image_paths)

		image_path = self.image_paths[real_idx]
		mask_path = self.mask_paths[real_idx]
		
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Swap image channels from BGR to RGB
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale mode

		# If this sample is for augmentation, apply additional transformations
		if is_augmented:
			image, mask = self.apply_augmentations(image, mask)
		
		# Apply the transformations to the image and mask needed by the model
		image = self.image_transforms(image)
		mask = self.mask_transforms(mask)
		
		return (image, mask)


	def apply_augmentations(self, image, mask):
		# Random horizontal flip with 50% probability
		if random.random() < 0.5:
			image = cv2.flip(image, 1)
			mask = cv2.flip(mask, 1)
		
		# Random vertical flip with 50% probability
		if random.random() < 0.5:
			image = cv2.flip(image, 0)
			mask = cv2.flip(mask, 0)
		
		# Gaussian blur or downscaling with 50% probability each
		if random.random() < 0.5:
			# Kernel size must be odd. Adjust (11,11) to control the strength of the blur.
			image = cv2.GaussianBlur(image, (11, 11), 0)
		else:
			# Get original image width and height
			original_width, original_height = image.shape[:2]
			# Calculate new downscaling size with factor of 0.3
			new_size = (int(original_height * 0.3), int(original_width * 0.3))
			# Downscale the image
			downscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
			# Upscale it back to the original dimensions
			image = cv2.resize(downscaled, (original_height, original_width), interpolation=cv2.INTER_LINEAR)
		
		# Random brightness adjustment with 50% probability
		if random.random() < 0.5:
			factor = 0.5 + random.random()  # Factor in the range [0.5, 1.5]
			image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
		
		return image, mask
