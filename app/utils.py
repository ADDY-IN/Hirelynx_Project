import re
import os
import fitz  # PyMuPDF
from docx import Document
from typing import List

# --- Text Extraction ---

def extract_text(file_path: str) -> str:
    """
    Detect resume file type and extract text.
    Supported formats: PDF, DOCX
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text).strip()
    elif extension == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()]).strip()
    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise ValueError("Unsupported format. Use PDF, DOCX, or TXT.")


# --- Text Cleaning & Tokenization ---

def clean_text(text: str) -> str:
    """
    Normalize text for better matching.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_text(text: str) -> List[str]:
    """
    Convert text into unique lowercase tokens.
    """
    return list(set(text.split()))

# --- JD Keyword Processing ---

STOPWORDS = {"and", "or", "the", "with", "for", "are", "looking", "experience"}

def extract_jd_keywords(description: str) -> List[str]:
    """
    Extract meaningful keywords from JD.
    """
    cleaned = clean_text(description)
    tokens = cleaned.split()
    return list(set([t for t in tokens if t not in STOPWORDS and len(t) > 2]))