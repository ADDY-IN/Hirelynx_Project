from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ScoringEngine:
    """
    Simplified Scoring Engine using Fuzzy Keyword Matching and TF-IDF.
    """
    def __init__(self, weight: float = 0.5):
        self.weight = weight
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def score(self, resume_text: str, jd_description: str, keywords: List[str]) -> Dict[str, Any]:
        # 1. Keyword Score (Fuzzy)
        tokens = list(set(resume_text.lower().split()))
        matched = [k for k in keywords if any(fuzz.ratio(k.lower(), t) > 80 for t in tokens)]
        kw_score = (len(matched) / len(keywords) * 100) if keywords else 0
        
        # 2. Semantic Score (TF-IDF)
        try:
            tfidf = self.vectorizer.fit_transform([resume_text, jd_description])
            sem_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100
        except:
            sem_score = 0

        final = round((kw_score * self.weight) + (sem_score * (1 - self.weight)), 2)
        
        return {
            "score": final,
            "matched_skills": matched,
            "recommendation": "Strong Match" if final > 80 else "Good Match" if final > 60 else "Potential Match" if final > 40 else "Low Match"
        }

