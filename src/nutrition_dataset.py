from torch.utils.data import Dataset
import torch

class NutritionDataset(Dataset):
    """
    A PyTorch Dataset class for training a model (LayoutLMv2) on nutrition OCR data extracted from images.


    Data Preparation Steps:

    For each row in the dataframe:
        1. Load the PIL image (column:processed_image)
        2. Extract words and bounding boxes from OCR tokens
        3. Convert BIO labels to integers using label_map
        4. Use the LayoutLMv2Processor to tokenize & align

    Args:
        dataframe must have the columns:
            processed_image: PIL.Image
            ocr_tokens: List of dicts, each with 'text' and 'bbox'
            bio_labels: List of (word, BIO-tag) tuples

        processor (LayoutLMv2Processor): Wraps tokenizer + feature extractor
        label_map: Mapping from BIO tags to IDs
    """
    
    def __init__(self, dataframe, processor, label_map):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get Row
        row = self.df.iloc[idx]

        # Extract image + Convert to RGB
        image = row['processed_image']
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract OCR info from tokens
        ocr_tokens = row['ocr_tokens']
        words = [token['text'] for token in ocr_tokens]
        boxes = [token['bbox'] for token in ocr_tokens]

        # BIO Lable Mapping
        bio_labels = row['bio_labels']
        labels = [self.label_map.get(label, 0) for _, label in bio_labels]

        # Processor : Tokenizes plus aligns tokens, bboxes, lablels
        # Automatically returns batched tensor: Fix next step
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Squeeze to remove batch dimension - since processing 1 image at a time
        # Dataloader will batch
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "labels": encoding["labels"].squeeze(0)
        }

        return item
