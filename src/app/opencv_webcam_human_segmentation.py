import sys
import cv2
import torch
import numpy as np
import config.config as cfg
from src.model.model_deeplabv3 import deeplabv3
from src.model.lightning_file import HumanSegmentation


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


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

model = deeplabv3()
checkpoint = "src/model/model_files/human-seg-epoch=12-validation_loss=0.06981.ckpt"
inference_model = HumanSegmentation.load_from_checkpoint(checkpoint_path=checkpoint, model=model)
inference_model.eval()
inference_model.to("cuda")

while cv2.waitKey(1) != 27:
    torch.cuda.empty_cache()

    ok, frame = source.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    # Prepare frame to be inputted into model
    input_image = cfg.IMAGE_TRANSFORMS(frame)
    input_image = input_image.unsqueeze(0).to("cuda") # create a mini-batch as expected by the model

    with torch.inference_mode():
        # Get model output logits
        logits = inference_model(input_image)['out'][0]  # Get the first item in the batch if in batch mode

        # Apply sigmoid to get probabilities, then threshold to get binary mask
        probabilities = torch.sigmoid(logits)
        binary_mask = probabilities > cfg.THRESHOLD

        # Convert binary mask to CPU and numpy array
        binary_mask = binary_mask.squeeze().cpu().numpy()  # Remove channel dimension

    # Resize the mask to fit the original frame dimensions
    original_width, original_height = frame.shape[:2]
    resized_mask = cv2.resize((binary_mask.astype(np.uint8)) * 255, (original_height, original_width), interpolation=cv2.INTER_NEAREST)

    # Overlay mask over image
    overlaid_image = overlay_mask(frame, resized_mask)
    
    cv2.imshow(win_name, overlaid_image)

    del input_image, logits, probabilities, binary_mask, resized_mask, overlaid_image

source.release()
cv2.destroyWindow(win_name)