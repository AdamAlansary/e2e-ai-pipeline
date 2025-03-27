import sys
import cv2
import torch
import numpy as np
import config.config as cfg
from src.model.model_deeplabv3 import deeplabv3
from src.model.lightning_file import HumanSegmentation


video_input_file_name = "data/test_videos/man_texting.mp4"


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


torch.cuda.empty_cache()

model = deeplabv3()
checkpoint = "src/model/model_files/human-seg-epoch=12-validation_loss=0.06981.ckpt"
inference_model = HumanSegmentation.load_from_checkpoint(checkpoint_path=checkpoint, model=model)
inference_model.eval()
inference_model.to("cuda")

# Read video
video = cv2.VideoCapture(video_input_file_name)

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = f"data/test_videos/man_texting_seg.mp4"
fps = video.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

i = 1
while True:
    torch.cuda.empty_cache()
    ok, frame = video.read()

    if not ok:
        break
    
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

    # Write frame to video
    video_out.write(overlaid_image)

    del input_image, logits, probabilities, binary_mask, resized_mask, overlaid_image

    print(f"Frame {i} processed")
    i+=1

video.release()
video_out.release()

# TODO: Need to train model on blurry and low quality images, since people that are not in focus are not recognized