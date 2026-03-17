from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ScoringEngine:
    """
    Advanced Scoring Engine using Fuzzy Keyword Matching and Sentence Embeddings.
    """
    def __init__(self, weight: float = 0.5):
        self.weight = weight
        try:
            logger.info("Loading SentenceTransformer model...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.encoder = None

    def score(self, resume_text: str, jd_description: str, keywords: List[str]) -> Dict[str, Any]:
        return self.score_with_embedding(resume_text, jd_description, None, keywords)

    def score_with_embedding(self, resume_text: str, jd_description: str, query_embedding: Any, keywords: List[str]) -> Dict[str, Any]:
        # 1. Keyword Score (Fuzzy Partial Ratio)
        resume_lower = resume_text.lower()
        matched = []
        
        if keywords:
            for k in keywords:
                if fuzz.partial_ratio(k.lower(), resume_lower) > 85:
                    matched.append(k)
                    
        kw_score = (len(matched) / len(keywords) * 100) if keywords else 0
        
        # 2. Semantic Score (Sentence Embeddings)
        sem_score = 0.0
        if self.encoder and resume_text:
            try:
                # Use pre-computed embedding if available, else compute it
                resume_embedding = self.encoder.encode([resume_text])[0]
                if query_embedding is None and jd_description:
                    target_embedding = self.encoder.encode([jd_description])[0]
                else:
                    target_embedding = query_embedding
                
                if target_embedding is not None:
                    sim = cosine_similarity(np.array([resume_embedding]), np.array([target_embedding]))[0][0]
                    sem_score = max(0.0, min(100.0, float(sim) * 100))
            except Exception as e:
                logger.error(f"Semantic scoring error: {e}")

        final = round((kw_score * self.weight) + (sem_score * (1 - self.weight)), 2)
        
        return {
            "score": final,
            "matched_skills": matched,
            "recommendation": "Strong Match" if final > 80 else "Good Match" if final > 60 else "Potential Match" if final > 40 else "Low Match"
        }

