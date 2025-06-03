# Nutrition OCR Extraction
This repo extracts structured nutritional information from nutrition label images using a fine-tuned LayoutLMv2 model.
It has both model fine tuning and inference hooks

## TRAINING
It uses OCR to get tokens and bounding boxes from images, converts to BIO tagging scheme, matches with labeled data (structured Nutriment dictionaries), fine tunes MicrosoftLMv2 on this data

## INFERENCE
It performs inference using fine-tuned transformer model trained with a BIO-tagging scheme to detect nutrient NAME, VALUE, and UNIT. The output is a clean, structured dictionary of nutriments with value and unit per nutrient in the dictionary.

## Features

- OCR token + bounding box extraction
- LayoutLMv2 for token classification
- BIO-tagging scheme (B/I for NAME, VAL, UNIT)
- End-to-end training pipeline
- End-to-end inference pipeline
- Postprocessing into JSON-style nutriment data

## Installation

```bash
pip install -r requirements.txt