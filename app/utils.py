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

STOPWORDS = {"and", "or", "the", "with", "for", "are", "looking", "experience", "in", "of", "to", "a"}

def extract_jd_keywords(description: str) -> List[str]:
    """
    Extract meaningful keywords/phrases from JD.
    Keeps consecutive non-stopwords as phrases (N-grams) for better semantic matching.
    """
    cleaned = clean_text(description)
    tokens = cleaned.split()
    
    keywords = set()
    current_phrase = []
    
    for token in tokens:
        if token in STOPWORDS or len(token) <= 2:
            if current_phrase:
                keywords.add(" ".join(current_phrase))
                current_phrase = []
        else:
            current_phrase.append(token)
            keywords.add(token) # also store the single word
            
    if current_phrase:
        keywords.add(" ".join(current_phrase))
        
    return list(set([k for k in keywords if len(k) > 2]))