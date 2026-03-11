# Hirelynx Resume Scorer & Parser

A complete resume scoring and parsing service built with Python. It compares resumes against job descriptions using keyword matching (Fuzzy Matching) and semantic similarity (TF-IDF).

## Features

- **Keyword Matching**: Fuzzy matching of JD keywords against resume tokens.
- **Semantic Similarity**: TF-IDF based cosine similarity for deeper content analysis.
- **Categorized Scoring**: Breakdown of how keywords and semantics contribute to the final score.
- **Schema Validation**: Maps parsed data to a standard `CandidateProfile` Pydantic model.
- **Workflow Automation**: S3-triggered event processing for both Resumes and JDs.
- **Batch Matching**: Batch processor to generate match scores across all profiles.
- **Recommendations**: Specialized functions for Candidate and Employer recommendations.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

## Usage

### 🚀 Running the Full Workflow Simulation
To see the complete S3 -> Parser -> Matching -> Recommendation flow:
```bash
python test_workflow.py
```

### Scoring a Local Resume (CLI)
```bash
python score_resume.py --resume path/to/resume.pdf --jd "Job description text here"
```

## Project Structure

- `app/`: Core logic modules.
  - `workflow.py`: S3 processors and recommendation logic.
  - `db_service.py`: Simulated database layer.
  - `scoring.py`: Enhanced scoring engine.
  - `resume_parser.py`: Pipeline coordinator.
  - `models.py`: Pydantic data models.
- `test_workflow.py`: Full system verification script.

