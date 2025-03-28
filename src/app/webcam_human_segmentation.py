import sys
import cv2
import torch
import src.segmentation.config as cfg
from src.segmentation.model_deeplabv3 import deeplabv3
from src.segmentation.utils import resize_mask, overlay_mask
from src.segmentation.lightning_file import HumanSegmentation


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

model = deeplabv3()
checkpoint = "models/human-seg-epoch=12-validation_loss=0.06981.ckpt"
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

        # Move binary mask to CPU and convert into numpy array
        binary_mask = binary_mask.squeeze().cpu().numpy()  # Remove channel dimension

    # Resize the mask to fit the original frame dimensions
    resized_mask = resize_mask(frame, binary_mask)

    # Overlay mask over image
    overlaid_image = overlay_mask(frame, resized_mask)
    
    cv2.imshow(win_name, overlaid_image)

source.release()
cv2.destroyWindow(win_name)