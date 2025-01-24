# Load the libraries

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Definitions

def show_batch_image(dataloader, index):

    # 1. Grab the image at the right index and convert to numpy array
    batch = next(iter(dataloader))
    numpy_array_img = batch[0][index].detach().numpy()
    numpy_array_img=np.transpose(numpy_array_img, (1, 2, 0))

    # 2. Plot image with label 

    image = Image.fromarray((255*numpy_array_img).astype(np.uint8))
    draw  = ImageDraw.Draw(image)
    draw.text((20,20), 
          dataloader.dataset.inv_labels.get(batch[1][index].item()), 
          fill="#FFFF00")
    
    plt.imshow(image)



def check_bw(DataLoader, type = "image"):

    image_keys = list(DataLoader.dataset.images.keys())
    image_paths = [DataLoader.dataset.images[key] for key in image_keys]
    images = [cv2.imread(path) for path in image_paths]

    annotated_images = []
    color = []

    for img in images:

        b, g, r = cv2.split(img)
        is_bw  = np.array_equal(b, g) and np.array_equal(g, r)
        bw_status = "BW" if is_bw else "Color"

        annotated_img = img.copy()
        cv2.putText(
            annotated_img, 
            bw_status, 
            (10, 30),  # Position: top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            (0, 255, 255),  # Yellow color (BGR format)
            2,  # Thickness
            cv2.LINE_AA  # Anti-aliased line
        )

        if type == "image":
            annotated_images.append(annotated_img)

        if type == "color":
            color.append(bw_status)

    if type == "image":
        return annotated_images
    
    if type == "color":
        return image_paths, color

