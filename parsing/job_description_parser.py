from parsing.file_loader import extract_text
from preprocessing.text_cleaner import clean_text

def parse_job_description(
    file_path: str | None = None,
    raw_text: str | None = None
) -> dict:
    """
    Extract raw text from a job description.

    Args:
        file_path (str | None): Path to JD file (PDF or DOCX)
        raw_text (str | None): Job description text input

    Returns:
        dict: {
            'source': 'job_description',
            'text': raw_text
        }
    """

    if not file_path and not raw_text:
        raise ValueError("Either file_path or raw_text must be provided")

    if file_path:
        text = extract_text(file_path)
    else:
        text = raw_text

    if not text or not text.strip():
        raise ValueError("Job description contains no readable text")

    return {
        "source": "job_description",
        "text": text
    }


parsed = parse_job_description('data/raw/job_description.docx')
cleaned_text = clean_text(parsed["text"])
print(cleaned_text)
