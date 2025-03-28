import cv2
import torch
import numpy as np
import src.segmentation.config as cfg
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from src.segmentation.model_deeplabv3 import deeplabv3
from src.segmentation.utils import resize_mask, overlay_mask
from src.segmentation.lightning_file import HumanSegmentation


model = deeplabv3()
checkpoint = "models/human-seg-epoch=12-validation_loss=0.06981.ckpt"
inference_model = HumanSegmentation.load_from_checkpoint(checkpoint_path=checkpoint, model=model)
inference_model.eval()
inference_model.to("cuda")

# Create FastAPI object
app = FastAPI(title="Human Segmentation API",
              description="API for segmenting humans in images",
              version="0.1.0")

# API operations
@app.get("/")
def health_check():
    return {"health_check": "OK"}


@app.post("/segment_image")
async def segment_image(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Prepare image to be inputted into model
    input_image = cfg.IMAGE_TRANSFORMS(image)
    input_image = input_image.unsqueeze(0).to("cuda") # create a mini-batch as expected by the model

    with torch.inference_mode():
        # Get model output logits
        logits = inference_model(input_image)['out'][0]

        # Apply sigmoid to get probabilities, then threshold to get binary mask
        probabilities = torch.sigmoid(logits)
        binary_mask = probabilities > cfg.THRESHOLD

        # Move binary mask to CPU and convert into numpy array
        binary_mask = binary_mask.squeeze().cpu().numpy()  # Remove channel dimension

    # Resize the mask to fit the original image dimensions
    resized_mask = resize_mask(image, binary_mask)

    # Overlay resized mask over image
    overlaid_image = overlay_mask(image, resized_mask)

    # Convert final image to bytes
    _, img_encoded = cv2.imencode('.png', overlaid_image)

    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/png")
