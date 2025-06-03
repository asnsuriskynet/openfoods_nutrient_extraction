from typing import Optional
from PIL import Image
import numpy as np
import cv2



'''
PSEUDOCODE:

FUNCTION process_image(pil_img):

    IF no image given:
        RETURN nothing

    TRY:
        Turn image to grayscale
        Remove noise from image
        Make image black and white
        Find how much the image is tilted
        Rotate image to straighten it
        Convert back to normal image format
        RETURN the processed image

    IF error happens:
        Show error message
        RETURN nothing
        
'''



def process_image(pil_img: Image.Image) -> Optional[Image.Image]:
    """
    Process a PIL Image by converting to grayscale, denoising,
    applying binary thresholding, and deskewing.

    Steps:
    1. Convert the input PIL image to grayscale (OpenCV format).
    2. Apply Non-local Means Denoising to reduce noise while preserving edges.
    3. Apply Otsu's binary thresholding to convert the denoised grayscale image to binary.
    4. Calculate the skew angle of the binary image using the minimum area rectangle around white pixels.
    5. Rotate (deskew) the binary image to correct skew.
    6. Convert the deskewed binary image back to a PIL Image and return.

    Args: Input image in PIL format.

    Returns: Processed PIL image after denoising,
                thresholding, and deskewing, or None if
                    rocessing fails or input is None.
    """

    if pil_img is None:
        return None

    try:
        # Grayscale
        if len(np.array(pil_img).shape) == 2:
            img_cv = np.array(pil_img)
        else:
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

        # Non-Local Means Denoising
        denoised_gray = cv2.fastNlMeansDenoising(img_cv, h=10)

        # Binary Thresholding
        _, binary = cv2.threshold(denoised_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Skew Calculation
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Deskew
        (h, w) = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(binary, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Convert and return
        final_img = Image.fromarray(deskewed)
        return final_img

    except Exception as e:
        print(f"Failed to process image: {e}")
        return None
