import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

class SummarizerService:
    def __init__(self):
        logger.info("Initializing Native Extractive Summarizer Service")

    def _extract_sections(self, text: str) -> dict:
        """Heuristically extracts key sections from Resume/JD text."""
        import re
        
        sections = {
            "Profile/Summary": [],
            "Experience/Responsibilities": [],
            "Skills/Requirements": [],
            "Education/Other": []
        }
        
        # Split text into rough lines or sentences. Split on single/double newlines and semicolons
        lines = re.split(r'\n+|\.\s+(?=[A-Z])|;\s*', text)
        
        current_section = "Profile/Summary"
        
        for line in lines:
            line_str = line.strip()
            # Ignore very short lines or lines that are ALL CAPS (often just noisy headers)
            if not line_str or len(line_str) < 15 or line_str.isupper():
                continue
                
            line_lower = line_str.lower()
            
            # Advanced keyword-based section routing
            if any(k in line_lower for k in ["experience", "responsibility", "what you'll do", "employment", "history", "role", "duties"]):
                current_section = "Experience/Responsibilities"
            elif any(k in line_lower for k in ["skill", "requirement", "qualifications", "technologies", "tech stack", "proficient", "knowledge"]):
                current_section = "Skills/Requirements"
            elif any(k in line_lower for k in ["education", "degree", "university", "college", "certification", "bachelor", "master"]):
                current_section = "Education/Other"
                
            # Add line to current section
            if len(sections[current_section]) < 4: # Keep sections brief (max 4 sentences)
                # Clean up weird characters common in parsing, but keep basic punctuation
                clean_line = re.sub(r'[^\w\s.,!?-]', ' ', line_str)
                clean_line = " ".join(clean_line.split()) # Remove multi-spaces
                if clean_line and len(clean_line) > 20: # Ignore short garbage lines
                    # Truncate extremely long paragraphs to keep it concise
                    if len(clean_line) > 200:
                        clean_line = clean_line[:197].rsplit(' ', 1)[0] + "..."
                    sections[current_section].append(clean_line)
                    
        return sections

    def summarize(self, text: str, max_length: int = 150) -> str:
        """
        Generates a structured, paragraph-based summary using pure Python logic.
        """
        if not text or len(text.strip()) < 50:
            return "Text is too short to generate a meaningful summary."

        # Limit input to avoid excessive regex processing
        clean_text = " ".join(text.split())[:8000] 
        
        sections = self._extract_sections(clean_text)
        
        formatted_summary = []
        
        # Format "Profile/Summary"
        if sections["Profile/Summary"]:
            formatted_summary.append("Overview: " + " ".join(sections["Profile/Summary"]))
            
        # Format "Experience/Responsibilities"
        if sections["Experience/Responsibilities"]:
            formatted_summary.append("Key Highlights & Experience: • " + " • ".join(sections["Experience/Responsibilities"]))
            
        # Format "Skills/Requirements"
        if sections["Skills/Requirements"]:
            formatted_summary.append("Core Skills & Requirements: • " + " • ".join(sections["Skills/Requirements"]))
            
        # Format "Education/Other"
        if sections["Education/Other"]:
            formatted_summary.append("Education & Background: • " + " • ".join(sections["Education/Other"]))

        if not formatted_summary:
            return "Failed to extract structured summary from text."

        return " | ".join(formatted_summary)

    def summarize_candidate_profile(self, candidate) -> str:
        """Generates a dynamic, first-person introduction using structural candidate data."""
        
        # Extract Personal Details
        name = "a professional"
        if candidate.personalDetails and isinstance(candidate.personalDetails, dict):
            name_val = candidate.personalDetails.get("name", "")
            if isinstance(name_val, str) and name_val:
                name = name_val.strip()
            
            # Fallback if name is empty but email exists
            if name == "a professional" or not name:
                email = candidate.personalDetails.get("email", "")
                if not email and candidate.resumeParsedJson:
                    email = candidate.resumeParsedJson.get("email", "")
                    
                if isinstance(email, list) and email:
                    email = email[0]
                if isinstance(email, str) and "@" in email:
                    import re
                    prefix = email.split("@")[0]
                    # Strip trailing numbers/symbols and capitalize
                    clean_prefix = re.sub(r'[^a-zA-Z]+$', '', prefix)
                    if clean_prefix:
                        # Add space between camelCase if present, else just title
                        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_prefix).title()
                        name = spaced

        # Extract Education
        education_str = ""
        if candidate.education and isinstance(candidate.education, list) and len(candidate.education) > 0:
            edu = candidate.education[0]
            degree = edu.get("degree", edu.get("degreeName", "a degree"))
            school = edu.get("school", edu.get("schoolName", edu.get("organization", "university")))
            
            if degree and degree != "a degree":
                education_str = f"I graduated with {degree} from {school}."
            else:
                education_str = f"I graduated from {school}."

        # Extract Work Experience / Current Role
        role_str = ""
        exp_details = []
        if candidate.workExperience and isinstance(candidate.workExperience, list):
            for exp in candidate.workExperience:
                title = exp.get("jobTitle")
                company = exp.get("companyName", exp.get("company", exp.get("organization")))
                
                if title and company and title != "N/A" and company != "Extracted from History":
                    if not role_str:
                        role_str = f"I have recently worked as a {title} at {company}."
                        
                resp = exp.get("responsibilities", [])
                if resp and isinstance(resp, list):
                    short_resp = str(resp[0])[:150]
                    if len(str(resp[0])) > 150:
                        short_resp += "..."
                    # Clean the responsibility string a bit
                    clean_resp = "".join([c for c in short_resp if c.isalnum() or c in " .,-"])
                    if clean_resp:
                        exp_details.append(clean_resp.strip())

        # Extract Skills
        skills_str = ""
        if candidate.skills and isinstance(candidate.skills, list):
            skill_names = []
            for s in candidate.skills[:5]: # Top 5 skills
                if isinstance(s, dict) and "name" in s:
                    skill_names.append(s["name"])
                elif isinstance(s, str):
                    skill_names.append(s)
            if skill_names:
                skills_str = f"My core technical skills include {', '.join(skill_names)}."

        # Bring it all together in a fluent bio
        bio_parts = [f"I am {name}."]
        if education_str: bio_parts.append(education_str)
        if role_str: bio_parts.append(role_str)
        if skills_str: bio_parts.append(skills_str)
        
        bio = " ".join(bio_parts)
        
        if exp_details:
            bio += f" Some highlights from my experience include: " + ". ".join(exp_details[:2]) + "."
            
        return bio

# Singleton instance
summarizer_service = SummarizerService()
