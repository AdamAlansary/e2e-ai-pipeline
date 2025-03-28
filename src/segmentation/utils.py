import cv2
import numpy as np


def resize_mask(image, mask):
    # Get image original width and height
    original_width, original_height = image.shape[:2]

    # Resize the mask to fit the original image dimensions
    resized_mask = cv2.resize((mask.astype(np.uint8)) * 255, (original_height, original_width), interpolation=cv2.INTER_NEAREST)

    return resized_mask


def overlay_mask(image, mask, alpha=0.5):
    # Create an image of the same size as the original image, filled with green
    green_overlay = np.zeros_like(image, dtype=np.uint8)
    green_overlay[:] = (0, 255, 0)  # OpenCV uses BGR format

    # Make a copy of the original image to overlay the mask
    overlaid_image = image.copy()

    # Create a boolean mask where the binary mask is active
    mask_bool = mask == 255

    # Only apply overlay if there are any True pixels in the mask
    if np.any(mask_bool):
        # For each pixel where mask is True, blend the original image with the green overlay only on the masked area
        overlaid_image[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha,
                                                     green_overlay[mask_bool], alpha, 0)
    else:
        # Optionally, log that no overlay is applied because no human was detected
        print("No active mask found; skipping overlay for this frame.")

    return overlaid_image
