import torch
import torch.nn as nn
from transformers import LayoutLMv2ForTokenClassification

class NutritionLayoutModel(nn.Module):
    def __init__(self, num_labels: int):
        
        """
        Initializes the LayoutLMv2 model for token classification tasks,
        customized to predict nutrition-related BIO labels.

        Args:
            num_labels (int): Number of distinct labels
        """
        super().__init__()
 
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv2-base-uncased",
            num_labels=num_labels
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        bbox: torch.Tensor, 
        labels: torch.Tensor = None
    ):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Token IDs for each token in the sequence (batch_size x seq_len).
            attention_mask (torch.Tensor): Mask to avoid attending to padding tokens.
            bbox (torch.Tensor): Bounding box coordinates for each token (batch_size x seq_len x 4).
            labels (torch.Tensor, optional): Ground-truth token labels for loss calculation.
        
        Returns:
            transformers.modeling_outputs.TokenClassifierOutput: Contains logits, loss (if labels provided), etc.
        """

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels
        )
