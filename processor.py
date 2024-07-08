import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
def pre_process(input_text):
    image_data = base64.b64decode(input_text)
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data))
    image = original_image.resize((32, 32), Image.Resampling.LANCZOS)
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to a numpy array
    img_array = np.array(image)
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    # Convert the numpy array to a Python list
    blocks_list = blocks.reshape(1, -1).tolist()[0]
    return blocks_list
