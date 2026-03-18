import re
import os
import fitz  # PyMuPDF
from docx import Document
from typing import List

# --- Text Extraction ---

def extract_text(file_path: str) -> str:
    """
    Extract text from resume files.
    Supports PDF, DOCX, TXT.
    Falls back to PDF if extension is missing.
    Raises explicit errors for unsupported or failed parsing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    # Fallback: assume PDF if no extension
    if not extension:
        extension = ".pdf"

    try:
        if extension == ".pdf":
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text_chunks.append(page_text)
            text = "\n".join(text_chunks).strip()

        elif extension == ".docx":
            doc = Document(file_path)
            text = "\n".join(
                para.text.strip()
                for para in doc.paragraphs
                if para.text and para.text.strip()
            ).strip()

        elif extension == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

        else:
            raise ValueError(f"Unsupported file format: {extension}")

        if not text:
            raise ValueError("Text extraction failed or document is empty.")

        return text

    except Exception as e:
        raise ValueError(f"Failed to extract text from {file_path}: {str(e)}")


# --- Text Cleaning & Tokenization ---

def clean_text(text: str) -> str:
    """
    Normalize text for better matching and remove database-breaking characters.
    """
    if not text:
        return ""
    # Remove null bytes and other non-printable characters that break Postgres JSON
    text = text.replace("\x00", "")
    # Remove surrogate pairs and other problematic unicode
    text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
    
    # Normalization for search/matching
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9+#./\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

def sanitize_for_db(text: str) -> str:
    """
    Remove only the absolutely necessary characters to prevent DB crashes,
    preserving as much original formatting as possible.
    """
    if not text:
        return ""
    # Postgres null byte fix
    text = text.replace("\x00", "")
    # Remove non-printable characters but keep common whitespace
    return "".join(c for c in text if c.isprintable() or c in "\n\r\t").strip()

# --- Skill & Keyword Refinement ---

SKILL_MAP = {
    "aws": "AWS",
    "python": "Python",
    "docker": "Docker",
    "react": "React",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "node": "Node.js",
    "fastapi": "FastAPI",
    "mysql": "MySQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "kubernetes": "Kubernetes",
    "jenkins": "Jenkins",
    "git": "Git",
    "linux": "Linux"
}

STOPWORDS = {"the", "and", "with", "from", "this", "that", "using", "for", "are", "looking", "experience", "in", "of", "to"}

def clean_keywords(words: List[str]) -> List[str]:
    """
    Clean and normalize matched skills/keywords.
    Filters out numbers, short words, and stopwords.
    """
    clean = []
    seen = set()
    for w in words:
        w_low = w.lower().strip()

        # remove numbers
        if re.search(r"\d", w_low):
            continue

        # remove short words
        if len(w_low) < 3:
            continue

        # remove stopwords
        if w_low in STOPWORDS:
            continue
            
        # Normalize via SKILL_MAP
        normalized = SKILL_MAP.get(w_low, w.strip())
        
        if normalized.lower() not in seen:
            clean.append(normalized)
            seen.add(normalized.lower())

    return clean

def tokenize_text(text: str) -> List[str]:
    """
    Convert text into unique lowercase tokens.
    """
    return list(set(text.split()))

# --- JD Keyword Processing ---

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