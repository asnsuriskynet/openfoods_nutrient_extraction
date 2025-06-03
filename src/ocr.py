from typing import List, Dict, Any
import pytesseract
from pytesseract import Output
import numpy as np


'''
PSEUDOCODE

FUNCTION extract_tokens_with_boxes(image):

    Run OCR on the image to get words, confidence, and bbox info

    FOR each detected word:
        IF word is not empty AND confidence > threshold:
            Get word text
            Get confidence score
            Get bounding box coordinates
            Save these in a list

    RETURN the list of words with their boxes and confidence
'''


def extract_tokens_with_boxes(img: np.ndarray, confidence_threshold: int) -> List[Dict[str, Any]]:
    """
    Extracts text tokens with corresponding bounding boxes + confidence scores from image 
    Uses Tesseract OCR.

    Steps:
    1. Perform OCR on the input image to get detailed data including detected text, confidence,
      and bounding box info.
    2. Iterate over each detected text token.
    3. Filter out empty tokens and those with low confidence
    4. For each valid token, collect its text, confidence score, and bounding box coordinates.
    5. Return a list of dictionaries, each containing:
       - 'text': the recognized string token,
       - 'conf': confidence score (integer),
       - 'bbox': bounding box in [left, top, right, bottom] format.

    Args: Input preprocessed image in OpenCV/numpy array format.

    Returns: List of dictionaries with keys 'text', 'conf', and 'bbox' for each recognized token.
    """
    data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng+fr')
    results: List[Dict[str, Any]] = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text and conf > confidence_threshold:
            bbox = [
                data['left'][i],
                data['top'][i],
                data['left'][i] + data['width'][i],
                data['top'][i] + data['height'][i]
            ]
            results.append({'text': text, 'conf': conf, 'bbox': bbox})

    return results

