import logging
from app.scoring import ScoringEngine

logging.basicConfig(level=logging.INFO)

def run_tests():
    print("Initializing Scoring Engine (this may take a moment to load the model)...")
    engine = ScoringEngine(weight=0.5)
    
    # Test Case 1: High Match
    print("\n--- Test Case 1: High Match ---")
    resume_text_1 = "Senior Machine Learning Engineer with 6 years of experience in Natural Language Processing, Python, and PyTorch. Built advanced scalable AI microservices on AWS."
    jd_desc_1 = "We are looking for an ML Engineer with strong Python skills and experience in NLP and deep learning frameworks like PyTorch."
    keywords_1 = ["machine learning", "python", "pytorch", "nlp", "aws"]
    
    result_1 = engine.score(resume_text_1, jd_desc_1, keywords_1)
    print(f"Keywords Matched: {result_1['matched_skills']}")
    print(f"Score: {result_1['score']}")
    print(f"Recommendation: {result_1['recommendation']}")

    # Test Case 2: Low Match (Semantic differentiation)
    print("\n--- Test Case 2: Low Match ---")
    resume_text_2 = "Frontend web developer. Expert in HTML, CSS, JavaScript, React, and building responsive UI designs for ecommerce platforms."
    jd_desc_2 = "We are looking for an ML Engineer with strong Python skills and experience in NLP and deep learning frameworks like PyTorch."
    keywords_2 = ["machine learning", "python", "pytorch", "nlp", "aws"]
    
    result_2 = engine.score(resume_text_2, jd_desc_2, keywords_2)
    print(f"Keywords Matched: {result_2['matched_skills']}")
    print(f"Score: {result_2['score']}")
    print(f"Recommendation: {result_2['recommendation']}")

if __name__ == "__main__":
    run_tests()
