from PIL import Image
import requests
from io import BytesIO



def download_image(url: str) -> Image.Image | None:
    """
    Downloads an image from a URL and returns it as a PIL Image object.
    Returns None if the download fails or the URL is invalid.
    """
    if isinstance(url, str) and url.startswith("http"):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Failed to fetch image: {e} | URL: {url}")
            return None
    return None


def extract_nutrition_url(image_dict: dict) -> str | None:
    """
    Extracts the first non-None URL from a dictionary where the key contains 'nutrition'.

    Parameters:
    - image_dict: dict or NaN

    Returns:
    - str (URL) or None
    """
    if isinstance(image_dict, dict):
        for key, value in image_dict.items():
            if 'nutrition' in key.lower() and value:
                return value
    return None