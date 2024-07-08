import numpy as np
from PIL import Image
from io import BytesIO
import base64

def pre_process(input):
    image_data = base64.b64decode(input)
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data))
    # Convert the image to a tensor
    image = original_image.resize((224, 224), Image.LANCZOS)
    # Convert the image to a numpy array and normalize it
    norm_img_data = np.array(image).astype('float32')
    norm_img_data = np.transpose(norm_img_data, [2, 0, 1])
    norm_img_data = np.expand_dims(norm_img_data, axis=0)
    return norm_img_data.tolist()
def post_process(input):
    output = np.array(input, dtype=np.float32)
    output = output.reshape(3, 224, 224)
    result = np.clip(output, 0, 255)
    result = result.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(result)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
