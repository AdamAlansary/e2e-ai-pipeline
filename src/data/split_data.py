# https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data
import os
import numpy as np
import pandas as pd
import src.segmentation.config as cfg

# Set seed for reproducibility
rng = np.random.default_rng(cfg.SEED)

df = pd.read_csv(cfg.CSV_PATH)
df = pd.DataFrame(df)

image_paths = np.array(list(map(lambda x: f"{cfg.BASE_PATH}/{x}", df['images'].values)))
mask_paths = np.array(list(map(lambda x: f"{cfg.BASE_PATH}/{x}", df['masks'].values)))

num_samples = len(image_paths)

train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# Shuffle indices using seed
indices = np.arange(num_samples)
rng.shuffle(indices)

# Calculate split boundaries
train_end = int(train_ratio * num_samples)
val_end = train_end + int(val_ratio * num_samples)

# Split indices for train, validation, and test sets
train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Assign image and mask paths for each split
train_images = image_paths[train_indices]
train_masks = mask_paths[train_indices]

val_images = image_paths[val_indices]
val_masks = mask_paths[val_indices]

test_images = image_paths[test_indices]
test_masks = mask_paths[test_indices]

# Create DataFrames and save new CSV files for each split
train_df = pd.DataFrame({'images': train_images, 'masks': train_masks})
val_df = pd.DataFrame({'images': val_images, 'masks': val_masks})
test_df = pd.DataFrame({'images': test_images, 'masks': test_masks})

train_df.to_csv(os.path.join(cfg.BASE_PATH, 'train.csv'), index=False)
val_df.to_csv(os.path.join(cfg.BASE_PATH, 'val.csv'), index=False)
test_df.to_csv(os.path.join(cfg.BASE_PATH, 'test.csv'), index=False)
