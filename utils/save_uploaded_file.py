import tempfile
import os

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file temporarily, preserving its extension,
    and return the file path.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name