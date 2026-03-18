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
    def extract_name(text: str) -> Dict[str, str]:
        """Extract first and last name from the first few lines."""
        all_lines = text.split('\n')
        lines = all_lines[:10] if len(all_lines) >= 10 else all_lines
        for line in lines:
            line = line.strip()
            # Basic heuristic: 2-3 words, no numbers, often at the start
            if 3 < len(line) < 30 and not re.search(r'\d', line):
                parts = line.split()
                if len(parts) >= 2:
                    return {"firstName": parts[0], "lastName": " ".join(parts[1:])}
        return {"firstName": "", "lastName": ""}

    @staticmethod
    def extract_location(text: str) -> Dict[str, str]:
        """Extract city and province/state."""
        # Simple list for Canada/India as requested in screens
        provinces = ["ontario", "bc", "quebec", "alberta", "up", "delhi", "haryana", "maharashtra", "karnataka"]
        cities = ["toronto", "vancouver", "ottawa", "ghaziabad", "noida", "delhi", "mumbai", "bangalore", "gurgaon"]
        
        found_city = ""
        found_province = ""
        
        text_low = text.lower()
        for city in cities:
            if re.search(rf'\b{re.escape(city)}\b', text_low):
                found_city = city.capitalize()
                break
        for prov in provinces:
            if re.search(rf'\b{re.escape(prov)}\b', text_low):
                found_province = prov.upper()
                break
        return {"city": found_city, "province": found_province}

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
        name_info = ResumeParser.extract_name(raw_text)
        email = ResumeParser.extract_email(raw_text)
        phone = ResumeParser.extract_phone(raw_text)
        loc_info = ResumeParser.extract_location(raw_text)
        
        # Skills from full text for better coverage
        skills = ResumeParser.extract_skills(cleaned)
        
        # Education from its specific section
        education = ResumeParser.extract_education(sections.get("education", ""))
        
        # Experience
        exp_text = str(sections.get("experience", ""))
        experience = []
        if exp_text:
            # Try to grab common fields
            experience.append(WorkExperience(
                companyName="Detected from History",
                role="Professional Experience",
                startDate="N/A",
                responsibilities=[s.strip() for s in exp_text.split('.') if len(s.strip()) > 10][:5]
            ))
            
        return {
            "personalDetails": PersonalDetails(
                firstName=name_info["firstName"],
                lastName=name_info["lastName"],
                phone=phone,
                location=loc_info["city"], # Mapping to 'city' or general location
                city=loc_info["city"],
                province=loc_info["province"]
            ).model_dump(),
            "skills": [s.model_dump() for s in skills],
            "education": [e.model_dump() for e in education],
            "workExperience": [w.model_dump() for w in experience],
            "email": email 
        }

parser = ResumeParser()
