import sys
import cv2
import torch
import src.segmentation.config as cfg
from src.segmentation.model_deeplabv3 import deeplabv3
from src.segmentation.utils import resize_mask, overlay_mask
from src.segmentation.lightning_file import HumanSegmentation


video_input_file_name = "tests/model/test_videos/bellingham.mp4"

model = deeplabv3()
# checkpoint = "models/human-seg-epoch=12-validation_loss=0.06981.ckpt"
checkpoint = "models/human_segmentation_model_v2.ckpt"
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

video_output_file_name = f"tests/model/test_videos/bellingham_seg.mp4" # TODO: Change to use input file name path and just add "_seg"
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

        # Move binary mask to CPU and convert into numpy array
        binary_mask = binary_mask.squeeze().cpu().numpy()  # Remove channel dimension

    # Resize the mask to fit the original frame dimensions
    resized_mask = resize_mask(frame, binary_mask)

    # Overlay mask over image
    overlaid_image = overlay_mask(frame, resized_mask)

    # Write frame to video
    video_out.write(overlaid_image)

    print(f"Frame {i} processed")
    i+=1

video.release()
video_out.release()
