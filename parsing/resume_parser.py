from parsing.file_loader import extract_text
from preprocessing.text_cleaner import clean_text

def parse_resume(file_path: str) -> dict:
    """
    Extract raw text from a resume file.

    Args:
        file_path (str): Path to the resume file (PDF or DOCX)

    Returns:
        dict: {
            'source': 'resume',
            'text': raw_text
        }
    """

    if not file_path:
        raise ValueError("Resume file path is required")

    raw_text = extract_text(file_path)

    if not raw_text.strip():
        raise ValueError("Resume file contains no readable text")

    return {
        "source": "resume",
        "text": raw_text
    }

parsed = parse_resume('../data/raw/CV-Jessy Gigato.pdf')
cleaned = clean_text(parsed['text'])