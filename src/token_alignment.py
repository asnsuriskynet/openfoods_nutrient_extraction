from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz
import re


'''
PSEUDOCODE

FUNCTION align_ocr_with_nutriments(ocr_tokens, nutriments, fuzzy_threshold, value_tol):

    Create a label list filled with 'O' for all tokens
    Keep track of tokens already labeled

    FOR each nutriment in nutriments:
        Clean the nutrient name
        Get Serving and Unit

        Find Tokens with fuzzy match to nutrients
        Label as 'B-NAME', following token are I-Name if they appear in nutrient name


        Find token that matches nutrient serving within tolerance
        Label as 'B-VAL'

        Find token that matches nutrient unit
        Label as 'B-UNIT'

    RETURN list of (token text, label) for all tokens
    
'''

def preprocess_token(text: str) -> str:
    """
    Lowercases and removes all non-alphanumeric characters from the input text.

    Args: The input string to preprocess.
    Returns: The cleaned, lowercase string with only alphanumeric characters.
    
    """
    return re.sub(r'[^a-z0-9]', '', text.lower())



def align_ocr_with_nutriments(ocr_tokens: List[Dict[str, Any]],
                              nutriments: List[Dict[str, Any]],
                              fuzzy_threshold: int = 85,
                              value_tol: float = 1.0) -> List[Tuple[str, str]]:
    
    """
    Aligns OCR tokens with nutriment data by labeling tokens as nutrient names,
    values, or units using fuzzy matching and numeric comparison.

    This function takes OCR-extracted tokens and attempts to match them against
    a list of nutriments (each having 'name', 'serving', and 'unit'). It assigns
    BIO-style labels ('B-NAME', 'I-NAME', 'B-VAL', 'B-UNIT', or 'O') to each token.

    Steps:
    1. Initialize labels as 'O' for all tokens.
    2. For each nutriment in the nutriments list:
       - Preprocess the nutrient name.
       - For each OCR token, preprocess and compare to nutrient name using fuzzy matching.
       - Label matching tokens as 'B-NAME' for the first and 'I-NAME' for continuation tokens.
       - Match tokens with numeric values close to the nutriment serving value within a tolerance.
       - Label such tokens as 'B-VAL'.
       - Match tokens exactly equal to the unit string.
       - Label such tokens as 'B-UNIT'.
    3. Ensure each token is labeled once (avoid re-labeling by tracking used indices).
    4. Return a list of tuples containing the original token text and its assigned label.

    Args:   List of OCR token dictionaries
            List of nutriment dicts
            Minimum fuzzy match score (0-100) for name matching. Defaults = 85.
            Numeric tolerance. Defaults to 1.0.

    Returns:   
            List of tuples where each tuple is (token_text, label).
            Labels follow BIO scheme: 'B-NAME', 'I-NAME', 'B-VAL', 'B-UNIT', 'O'.
    """

    labels = ['O'] * len(ocr_tokens)
    used_indices = set()

    for nutrient in nutriments:
        name = preprocess_token(nutrient.get('name', ''))
        serving = nutrient.get('serving')
        unit = nutrient.get('unit')

        # Nutrient Name Fuzzy Match
        for i, token in enumerate(ocr_tokens):
            token_text = preprocess_token(token['text'])
            if i in used_indices or not token_text:
                continue
            score = fuzz.partial_ratio(name, token_text)
            if score >= fuzzy_threshold:
                labels[i] = 'B-NAME'
                used_indices.add(i)

                # Label continuation labelling
                j = i + 1
                while j < len(ocr_tokens):
                    next_text = preprocess_token(ocr_tokens[j]['text'])
                    if next_text and next_text in name:
                        labels[j] = 'I-NAME'
                        used_indices.add(j)
                        j += 1
                    else:
                        break

        # Nutrient Value Serving Match (Numeric)
        if serving is not None:
            for i, token in enumerate(ocr_tokens):
                if i in used_indices:
                    continue
                try:
                    if abs(float(token['text']) - float(serving)) < value_tol:
                        labels[i] = 'B-VAL'
                        used_indices.add(i)
                except ValueError:

                    continue

        # Nutrient Unit match
        if unit:
            for i, token in enumerate(ocr_tokens):
                token_text = preprocess_token(token['text'])
                if i in used_indices:
                    continue
                if token_text == preprocess_token(unit):
                    labels[i] = 'B-UNIT'
                    used_indices.add(i)

    return [(token['text'], labels[i]) for i, token in enumerate(ocr_tokens)]


def apply_alignment(row):
    ocr = row['ocr_tokens']
    nutr = row['nutriments']
    return align_ocr_with_nutriments(ocr, nutr)
