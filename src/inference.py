import torch
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from transformers import LayoutLMv2Processor
from PIL import Image
from src.nutrition_model import NutritionLayoutModel
from src.utils import download_image, extract_nutrition_url
from src.image_preprocessing import process_image
from src.ocr import extract_tokens_with_boxes
from src.token_alignment import apply_alignment



'''
PSEUDOCODE:

1. Parse command-line arguments:
    - image_path: path to the nutrition label image
    - model_path: path to the saved model (.pth file)

2. Set device (GPU/CPU)
3. Define label_map
4. Load:
    - LayoutLMv2 processor
    - Previously Trained LayoutLMv2 model using load_model()

5. Run prediction:
    a. Load image and preprocess it with processor
    b. Move tensors to the appropriate device
    c. Perform inference with model
    d. Convert logits to predicted label indices
    e. Map indices to label strings
    f. Convert input IDs to tokens
    g. Zip tokens with predicted labels

6. Post-process token-label pairs into structured nutriments dictionary
7. Print results

'''



def load_model(model_path: str, num_labels: int, device: torch.device) -> NutritionLayoutModel:
    """
    Load the trained NutritionLayoutModel from a checkpoint.

    Args:
        model_path (str): Path to the saved model .pth file.
        num_labels (int): Number of output classes.
        device (torch.device): Device to load the model on (CPU/GPU).

    Returns:
        NutritionLayoutModel: Loaded model set to evaluation mode.
    """
    model = NutritionLayoutModel(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def prepare_input(image: Image.Image, processor: LayoutLMv2Processor) -> Dict[str, torch.Tensor]:
    """
    Preprocess a PIL image and convert it into model input format including tokens and bounding boxes.

    Args: 
        The input image to preprocess.
        processor (LayoutLMv2Processor): The HuggingFace processor for LayoutLMv2.

    Returns:
        Dictionary containing input_ids, attention_mask, and bbox tensors.
    """
    # Preprocess image
    processed_image = process_image(image)

    # Extract OCR tokens and bounding boxes
    ocr_tokens = extract_tokens_with_boxes(processed_image)

    # Extract words and bounding boxes from OCR tokens
    words = [token["text"] for token in ocr_tokens]
    boxes = [token["box"] for token in ocr_tokens]

    # Encode inputs for LayoutLMv2
    encoding = processor(
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return encoding

def extract_nutriments(token_label_pairs: List[Tuple[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert token-label pairs into structured nutriment information.

    Args:
        token_label_pairs: List of (token, label) tuples from model prediction.

    Returns:
        Dictionary with key 'nutriments' and index-mapped structured nutrition fields.
    """
    nutriments = {}
    current_item = {"name": "", "value": "", "unit": ""}
    current_tag = None
    idx = 0

    for token, label in token_label_pairs:
        if token in ["[PAD]", "[CLS]", "[SEP]"]:
            continue  #Skip Special Tokens

        token = token.replace("▁", "").strip()

        if label == "O":
            continue  # skip tokens outside entities

        label_type = label.split("-")[-1]

        if label.startswith("B-"):
            if any(current_item.values()):
 
                nutriments[idx] = {
                    "name": current_item["name"].strip(),
                    "unit": current_item["unit"].strip() or None,
                    "value": try_parse_float(current_item["value"])
                }
                idx += 1
                current_item = {"name": "", "value": "", "unit": ""}

            current_item[label_type.lower()] = token

        elif label.startswith("I-"):
            if current_item[label_type.lower()] != "":
                current_item[label_type.lower()] += " " + token
            else:
                current_item[label_type.lower()] = token


    if any(current_item.values()):
        nutriments[idx] = {
            "name": current_item["name"].strip(),
            "unit": current_item["unit"].strip() or None,
            "value": try_parse_float(current_item["value"])
        }

    return {"nutriments": nutriments}


def try_parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except ValueError:
        return None



def predict(image_path: str,
            model: NutritionLayoutModel,
            processor: LayoutLMv2Processor,
            label_map: Dict[str, int],
            device: torch.device) -> List[Tuple[str, str]]:
    """
    Perform token classification inference on a nutrition label image.
    Return restructured data

    Args:
        Path to the input nutrition label image.
        The trained LayoutLMv2 model for token classification.
        Processor for tokenizing and encoding inputs.
        Mapping of label names to their integer IDs.
        Device to run inference on (CPU or GPU).

    Returns:
        List of tuples where each tuple contains a token and its predicted label.
    """

    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    encoding = prepare_input(image, processor)

    # Tensors to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    bbox = encoding["bbox"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox
        )
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    # Map prediction indices back to label names
    id_to_label = {v: k for k, v in label_map.items()}
    predicted_labels = [id_to_label.get(idx, "O") for idx in predictions]

    # Convert input ids back to tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())

    # Combine tokens and predicted labels into tuples
    token_label_pairs = list(zip(tokens, predicted_labels))

    # Extract nutriments from token_label_pairs
    nutriments = extract_nutriments(token_label_pairs)

    return token_label_pairs, nutriments



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for Nutrition LayoutLMv2 model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to nutrition label image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2, 'B-VAL': 3, 'I-VAL': 4, 'B-UNIT': 5, 'I-UNIT': 6}
    
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    model = load_model(args.model_path, num_labels=len(label_map), device=device)

    # Predict
    token_labels, nutriments = predict(args.image_path, model, processor, label_map, device)

    # Print Predictions
    print("\nPredicted tokens and labels:")
    for token, label in token_labels:
        print(f"{token}: {label}")

    print('Nutriments: ', nutriments)
