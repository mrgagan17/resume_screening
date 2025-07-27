import os
from pdfminer.high_level import extract_text
import docx2txt

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text(path)
    elif ext in ['.docx', '.doc']:
        return docx2txt.process(path)
    else:
        return ''

def save_uploaded_file(file, upload_dir):
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, file.filename)
    file.save(path)
    return path
