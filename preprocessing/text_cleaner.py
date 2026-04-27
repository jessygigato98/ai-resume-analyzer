import re

def _normalize_whitespace(text: str) -> str:
    """
    Replace multiple spaces and line breaks with a single space.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _remove_non_printable_chars(text: str) -> str:
    """
    Remove non-printable or control characters.
    """
    return re.sub(r"[^\x20-\x7E]", " ", text)


def _basic_text_cleanup(text: str) -> str:
    """
    Basic text cleanup without removing semantic content.
    """
    # Lowercase
    text = text.lower()

    # Remove strange bullet characters
    text = re.sub(r"[•◦▪●■]", " ", text)

    # Normalize whitespace
    text = _normalize_whitespace(text)

    return text


def clean_text(raw_text:str) -> str: 
    """
        Clean raw extracted text to prepare it for embeddings.

        Args:
            raw_text (str): Raw extracted text

        Returns:
            str: Cleaned text
    """
    if not raw_text or not raw_text.strip():
            raise ValueError("Input text is empty or invalid")

    text = raw_text
    text = _remove_non_printable_chars(text)
    text = _basic_text_cleanup(text)

    return text
