import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from src.nutrition_dataset import NutritionDataset
from src.nutrition_model import NutritionLayoutModel
from src.utils import download_image, extract_nutrition_url
from src.image_preprocessing import process_image
from src.ocr import extract_tokens_with_boxes
from src.token_alignment import preprocess_token,align_ocr_with_nutriments, apply_alignment

from transformers import LayoutLMv2Processor


'''
PSEUDOCODE

MAIN:
-----
Load dataframe from path

For each row in the dataframe:
    - Extract the nutrition image URL
    - Download the image from the URL
    - Preprocess the image (resize, normalize, etc.)
    - Extract tokens and bounding boxes using Tesseract OCR
    - Align OCR tokens to BIO labels

Initialize:
    - LayoutLMv2Processor from HuggingFace
    - label_map for all entity tags (B/I/O for NAME, VALUE, UNIT)

Split dataframe into training and validation sets (80/20)

Wrap train/val data in NutritionDataset (custom Pytorch Dataset class)
Create PyTorch DataLoaders for batching

Initialize LayoutLMv2 model with correct number of labels
Move model to GPU if available

Create optimizer (AdamW)
Create scheduler (linear warmup/decay)
Define number of training epochs

Call `train()` with model, dataloaders, optimizer, scheduler, etc.



TRAIN:
------
Set model to train mode

For each epoch in total_epochs:
    - Initialize epoch loss to 0
    - Loop over training batches:
        - Move inputs to device
        - Forward pass: compute outputs and loss
        - Zero gradients
        - Backward pass: compute gradients
        - Optimizer step
        - Scheduler step
        - Accumulate loss
        - Update progress bar

    Compute average training loss for the epoch
    Call `validate()` to compute validation loss, accuracy, precision, recall, F1
    Print all metrics
    Save model weights to checkpoint file

VALIDATE:
------

Set model to evaluation mode

For each batch in validation dataloader:
    - Move batch to device
    - Run model forward pass without gradients
    - Compute loss
    - Get logits and derive predictions using argmax
    - Mask out padding tokens (-100) and flatten predictions
    - Accumulate predictions and true labels

Compute average validation loss

Use sklearn to compute:
    - Accuracy
    - Precision
    - Recall
    - F1 score (weighted)

Return metrics dictionary
Set model back to training mode


'''

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    label_map: dict
) -> None:
    """
    Train the LayoutLMv2 model for token classification.

    Args:
        The PyTorch model to train.
        Train DataLoader that provides batches of training data.
        Optimizer for model parameters.
        Learning rate scheduler.
        Device to run training on (CPU/GPU).
        Number of full passes over the training data.

    Returns:
        None
    """
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass -- forward called implicity
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                labels=labels
            )
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        val_metrics = validate(model, val_dataloader, device, label_map)
        print(
            f"Validation Loss: {val_metrics['loss']:.4f} | "
            f"Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1 Score: {val_metrics['f1']:.4f}"
        )

        save_path = f"./checkpoints/nutrition_model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")



def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_map: dict
) -> dict:
    """
    Run model evaluation on validation set.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for validation dataset.
        device: Device (CPU/GPU) to run evaluation on.
        label_map: Dictionary mapping label names to indices.

    Returns:
        metrics (dict): Dictionary containing loss, accuracy, precision, recall, f1.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Flatten and mask padding tokens (-100 is padding)
            active_mask = labels != -100
            preds = preds[active_mask].cpu().numpy()
            labels = labels[active_mask].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    model.train()
    return metrics



def main():
    # Load Dataframe
    df = pd.read_parquet("./data/df_experimental.parquet")

    # Extract Nutrition Images
    df['nutrition_url'] = df['image_urls'].apply(extract_nutrition_url)
    df['nutrition_image'] = df['nutrition_url'].apply(download_image)

    # Apply image preprocessing
    df['processed_image'] = df['nutrition_image'].apply(process_image)

    # Extract tokens, boxes with Tesserac
    df['ocr_tokens'] = df['processed_image'].apply(extract_tokens_with_boxes)

    # Drop rows with no tokens
    df = df[df['ocr_tokens'].astype(bool)]

    # Align OCR Tokens to BIO labels
    df['bio_labels'] = df.apply(apply_alignment, axis=1)
    
    # Initialize processor and label_map
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    label_map = {'O': 0, 'B-NAME': 1, 'I-NAME': 2, 'B-VAL': 3, 'I-VAL': 4, 'B-UNIT': 5, 'I-UNIT': 6}

    # Train Test Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Create dataset and dataloader
    train_dataset = NutritionDataset(train_df, processor, label_map)
    val_dataset = NutritionDataset(val_df, processor, label_map)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model
    model = NutritionLayoutModel(num_labels=len(label_map))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Set up scheduler
    num_epochs = 3
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Call Loop and Train model
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, label_map)

if __name__ == "__main__":
    main()
