# Load the libraries

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Definitions

def show_batch_image(batch, index):

    # 1. Grab the image at the right index and convert to numpy array
    
    numpy_array_img = batch[0][index].detach().numpy()
    numpy_array_img=np.transpose(numpy_array_img, (1, 2, 0))

    # 2. Plot image with label 

    image = Image.fromarray((255*numpy_array_img).astype(np.uint8))
    draw  = ImageDraw.Draw(image)
    draw.text((20,20), 
          dl_test.dataset.inv_labels.get(batch[1][index].item()), 
          fill="#FFFF00")
    
    plt.imshow(image)

