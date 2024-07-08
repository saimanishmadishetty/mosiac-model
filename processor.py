import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

def pre_process(input):
    #print("Input to pre_transform (first 20 characters):", input[:20])
    image_data = base64.b64decode(input)
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data))
    # Convert the image to a tensor
    image = original_image.resize((224, 224))
    # Convert the image to a numpy array and normalize it
    image_np = np.array(image)
    image_np = image_np.astype(np.uint8)
    # Convert to list
    image_list = image_np.tolist()
    return image_list

def post_process(input):
    predictions = input
    boxes = np.array(predictions['detection_boxes'])
    scores = np.array(predictions['detection_scores'])
    classes = np.array(predictions['detection_classes'])
    num_detections = int(predictions['num_detections'])
    
    # Create a boolean mask where scores are greater than 0.5
    mask = scores > 0.5
    
    # Use the mask to filter the boxes, scores, and classes
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_classes = classes[mask]
    
    filtered_predictions = {
        'detection_boxes': filtered_boxes.tolist(),
        'detection_scores': filtered_scores.tolist(),
        'detection_classes': filtered_classes.tolist(),
        'num_detections': len(filtered_scores)
    }
    
    return filtered_predictions
