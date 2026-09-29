import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    # Pattern: capture delimiters as separate tokens
    # Delimiters: whitespace, individual punctuation , . : ; ? _ ! " ( ) '
    # and the double-dash sequence --
    pattern = r'(--|[\s,.:;?_!"()\'])'

    parts = re.split(pattern, text)
    tokens = [p.strip() for p in parts if p is not None and p.strip() != '']
    return tokens