# Hirelynx AI Backend

Professional-grade AI recruitment backend for resume parsing, job matching, and candidate intelligence. Built with FastAPI and powered by Groq LLMs.

## 🚀 Overview

Hirelynx is a high-performance microservice that automates the recruitment pipeline. It uses advanced LLMs (Llama 3) and semantic search to bridge the gap between candidate profiles and job requirements.

## 🏗️ New Modular Architecture

The project has been restructured for professional maintainability and scaling:

```text
app/
├── main.py                 # Slim entry point (FastAPI initialization)
├── api/                    # API Layer (Versioned)
│   └── v1/                 # Domain-specific routers
│       ├── candidate.py    # Bio generation & Candidate tools
│       ├── recruiter.py    # Job summaries & NOC duties
│       ├── employer.py     # Company profile generation
│       ├── admin.py        # Smart candidate search
│       ├── parser.py       # Resume indexing
│       └── scoring.py      # Match scoring engine
├── services/               # Service Layer (Business Logic)
│   ├── summarizer/         # Decomposed summarization logic
│   ├── parser.py           # Resume parsing engine
│   ├── scoring.py          # LLM-powered matching engine
│   ├── search_service.py   # Vector & Filter search
│   └── workflow.py         # Orchestration & Pipeline logic
├── models/                 # Data Layer (SQLAlchemy Models)
├── schemas/                # Data Layer (Pydantic Schemas)
├── core/                   # Core Infrastructure (Config, Utils)
└── db/                     # Database Session Management
```

## ✨ Key Features

- **AI Resume Parsing**: Extracts structured data from PDF/Docx using Groq LLM.
- **Semantic Job Matching**: Scores candidates against JDs with detailed breakdowns.
- **Smart Candidate Search**: Hybrid search (Filter + AI semantic ranking).
- **Personalized Content**: Generates "About Me" bios and tailored Job Responsibilities.
- **NOC Integration**: Automatically personalizes standard NOC duties for specific companies.
- **Company Branding**: Scrapes websites to generate high-impact employer profiles.

## 🛠️ Installation & Setup

1. **Clone the repository**
2. **Setup environment**
   ```bash
   cp .env.example .env  # Add your GROQ_API_KEY and AWS credentials
   ```
3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```
4. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

## 🧪 Testing

- **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Workflow Simulation**: `python test_workflow.py`
