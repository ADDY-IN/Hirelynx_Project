import re
from typing import List, Dict, Any, Optional
from app.models import PersonalDetails, Education, WorkExperience, Skill, Project

class ResumeParser:
    # Expanded skill list with normalization mapping if needed, 
    # but for now we just use upper() as requested.
    SKILL_DB = [
        "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust",
        "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "cassandra", "elasticsearch",
        "fastapi", "flask", "django", "nodejs", "react", "angular", "vue", "nextjs", "express", "spring boot",
        "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "jenkins", "git", "github", "linux",
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
        "rest api", "graphql", "kafka", "rabbitmq", "microservices", "unit testing", "ci/cd", "agile",
        "devops", "cicd", "shell", "bash", "cybersecurity", "compliance", "sonar qube", "fortify", "jira"
    ]

    @staticmethod
    def clean_resume_text(text: str) -> str:
        """
        Aggressive cleaning for parsing logic.
        """
        # Remove extra symbols but keep email/phone related ones
        text = re.sub(r"[^\w\s@.+:/-]", " ", text)
        # Normalize spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def split_sections(text: str) -> Dict[str, str]:
        """
        Core Fix: Capture content between headers using non-greedy regex.
        """
        sections = {}
        patterns = {
            "education": r"education(.*?)(experience|projects|skills|summary|contact|$)",
            "experience": r"experience(.*?)(education|projects|skills|summary|contact|$)",
            "projects": r"projects(.*?)(education|experience|skills|summary|contact|$)",
            "skills": r"skills(.*?)(education|experience|projects|summary|contact|$)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.I | re.S)
            if match:
                sections[key] = match.group(1).strip()
        return sections

    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)
        return match.group(0) if match else None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        # Match 10-digit number
        match = re.search(r"\b\d{10}\b", text)
        return match.group(0) if match else None

    @staticmethod
    def extract_location(text: str) -> Optional[str]:
        # User requested specific location regex
        match = re.search(r"(ghaziabad|delhi|mumbai|bangalore|noida|gurgaon)[, ]*(up|india|haryana|maharashtra)?", text, re.I)
        return match.group(0) if match else None

    @staticmethod
    def extract_skills(text: str) -> List[Skill]:
        text_lower = text.lower()
        found_skills = set()
        for skill in ResumeParser.SKILL_DB:
            pattern = rf'(?i)\b{re.escape(skill)}\b'
            if re.search(pattern, text_lower):
                # Normalize to Upper as requested
                found_skills.add(skill.upper())
        return [Skill(name=s, level="Found") for s in sorted(list(found_skills))]

    @staticmethod
    def extract_education(text: str) -> List[Education]:
        """
        Targeted extraction for education levels.
        """
        if not text:
            return []
            
        degrees = {
            "mca": "Master of Computer Applications",
            "bca": "Bachelor of Computer Applications",
            "btech": "B.Tech",
            "mtech": "M.Tech",
            "bachelor": "Bachelor's Degree",
            "master": "Master's Degree",
            "secondary": "Secondary Education"
        }
        
        results = []
        text_lower = text.lower()
        
        for code, full_name in degrees.items():
            if code in text_lower:
                results.append(Education(
                    degree=full_name,
                    # For institution, we try to take the text around it if it's not the header
                    institution="Detected in record"
                ))
        return results

    @staticmethod
    def parse(raw_text: str) -> Dict[str, Any]:
        """
        Production Parser: Segment -> Extract -> Structure.
        """
        # 1. Clean
        cleaned = ResumeParser.clean_resume_text(raw_text)
        
        # 2. Segment
        sections = ResumeParser.split_sections(cleaned)
        
        # 3. Extract Fields
        email = ResumeParser.extract_email(raw_text) # Use raw for email/phone to avoid too much cleaning
        phone = ResumeParser.extract_phone(raw_text)
        location = ResumeParser.extract_location(raw_text)
        
        # Skills from full text for better coverage
        skills = ResumeParser.extract_skills(cleaned)
        
        # Education from its specific section
        education = ResumeParser.extract_education(sections.get("education", ""))
        
        # Experience (Level 2: Basic segmentation for now)
        exp_text = str(sections.get("experience", ""))
        experience = []
        if exp_text:
            experience.append(WorkExperience(
                companyName="Extracted from History",
                startDate="N/A",
                responsibilities=[exp_text[:500] + "..."] # Capture snippet
            ))
            
        # Projects
        proj_text = str(sections.get("projects", ""))
        projects = []
        if proj_text:
            projects.append(Project(
                title="Extracted Project",
                summary=proj_text[:500] + "..."
            ))

        return {
            "personalDetails": PersonalDetails(
                phone=phone,
                location=location
            ).model_dump(),
            "skills": [s.model_dump() for s in skills],
            "education": [e.model_dump() for e in education],
            "workExperience": [w.model_dump() for w in experience],
            "projects": [p.model_dump() for p in projects],
            "email": email 
        }

parser = ResumeParser()
