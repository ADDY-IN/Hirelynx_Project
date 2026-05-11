"""
Resume Parser — Groq-First
==========================
Primary extraction: Groq LLM (llama-3.3-70b-versatile)
Fallback:          Regex for email + phone only

~150 lines. No NER. No blocklists. No scoring.
"""
import re
import json
import logging
from typing import Any, Dict, List, Optional

from app.models import PersonalDetails, Education, WorkExperience, Skill, Project

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client — loaded lazily, once
# ---------------------------------------------------------------------------
_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    try:
        from groq import Groq
        from app.core.config import settings
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — parser will return empty fields.")
            return None
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        return _groq_client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


# ---------------------------------------------------------------------------
# The single prompt — everything extracted in one call
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a precise resume data extractor. 
Extract information from resumes and return ONLY valid JSON — no markdown, no explanation, no code blocks.
If a field is missing or unclear, use an empty string "" or empty array [].
Never invent or assume information not present in the resume."""

_USER_PROMPT_TEMPLATE = """Extract resume data and return this exact JSON structure:

{{
  "firstName": "",
  "lastName": "",
  "phone": "",
  "city": "",
  "province": "",
  "country": "",
  "workType": "",
  "skills": [],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "fieldOfStudy": "",
      "startDate": "",
      "endDate": ""
    }}
  ],
  "workExperience": [
    {{
      "companyName": "",
      "role": "",
      "startDate": "",
      "endDate": "",
      "currentlyWorking": false,
      "responsibilities": []
    }}
  ],
  "projects": [
    {{
      "title": "",
      "tools": [],
      "startDate": "",
      "summary": ""
    }}
  ],
  "certifications": [
    {{
      "name": "",
      "issuer": "",
      "issueDate": ""
    }}
  ],
  "summary": ""
}}

Rules:
- firstName / lastName: candidate's real name only — NEVER a city, job title, or address
- skills: list every skill/technology mentioned, as short strings (e.g. "Python", "React", "SQL")
- workType: one of "REMOTE", "HYBRID", "ON_SITE", or "" if not mentioned
- currentlyWorking: true only if "present", "current", or "ongoing" appears in that job entry
- province: use the 2-letter abbreviation for Canadian provinces (ON, BC, AB, QC, etc.)
- summary: first 2-3 sentences summarizing the candidate's professional profile
- projects.startDate: capture the month/year shown next to the project (e.g. "March-23", "Oct-22")
- projects.summary: use the one-liner description written below the project title
- certifications: ONLY include formal professional certifications or courses (e.g. AWS Certified Developer, Google Analytics Certificate). Do NOT include achievements, awards, medals, prizes, competition ranks, or LeetCode/HackerRank scores — those are NOT certifications.
- workExperience: Capture the FULL professional history. Extract every job, role, and set of responsibilities. DO NOT skip any entries.
- workExperience.responsibilities: Extract ALL bullet points or duty descriptions for each role as a list of strings. Be thorough.
- Return empty string "" for any missing field, NOT null

RESUME TEXT:
{raw_text}"""


# Regex helpers 

def _extract_email(text: str) -> Optional[str]:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    """Extract phone number. Requires 8+ digits. Rejects date patterns like 2021-2023."""
    # Reject pure year-range patterns first
    cleaned = re.sub(r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}\b", "", text)
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)  # remove standalone years too

    # Try international format
    m = re.search(
        r"(\+?\d{1,3}[\s\-.])?(\(?\d{2,4}\)?[\s\-.])?(\d{3,5}[\s\-\.]\d{3,5}([\s\-\.]\d{2,4})?)",
        cleaned,
    )
    if m and len(re.sub(r"\D", "", m.group(0))) >= 8:
        return m.group(0).strip()
    # 10-digit solid number fallback
    m = re.search(r"\b\d{10}\b", cleaned)
    return m.group(0) if m else None

# Main parser

def _call_groq(raw_text: str) -> Dict[str, Any]:
    """Send resume text to Groq, return parsed dict."""
    client = _get_groq_client()
    if not client:
        return {}

    # Cap text to keep within model context
    # Projects appear late in resumes, so we use 12000 chars.
    # For very long resumes, preserve both the start AND the end
    # so skills/experience AND projects/certs are both captured.
    MAX_CHARS = 12000
    if len(raw_text) <= MAX_CHARS:
        text_chunk = raw_text
    else:
        # Take first 8000 (personal info, skills, experience) +
        # last 4000 (projects, certifications, education at end)
        text_chunk = raw_text[:8000] + "\n...\n" + raw_text[-4000:]
    prompt = _USER_PROMPT_TEMPLATE.format(raw_text=text_chunk)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,       # deterministic
            max_tokens=4096,       # enough for dense resumes (30+ skills, 8+ projects)
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Groq parsing failed (Rate Limit/Down). Routing to Gemini: {e}")
        
    # ── Fallback to Gemini ───────────────────────────────────────────────
    # Only gemini-3-flash-preview has a working quota on this account.
    try:
        import google.generativeai as genai
        from app.core.config import settings
        gemini_api_key = getattr(settings, "GEMINI_API_KEY", None)
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not set — cannot use Gemini fallback.")
            return {}

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')

        # Use the same smart-chunked text as Groq for identical coverage
        gemini_prompt = f"""Extract resume data and return ONLY a JSON object. No markdown. No explanation.

JSON format:
{{"firstName":"","lastName":"","phone":"","city":"","province":"","country":"","workType":"","summary":"","skills":[],"education":[{{"degree":"","institution":"","fieldOfStudy":"","startDate":"","endDate":""}}],"workExperience":[{{"companyName":"","role":"","startDate":"","endDate":"","currentlyWorking":false,"responsibilities":[]}}],"projects":[{{"title":"","tools":[],"startDate":"","summary":""}}],"certifications":[{{"name":"","issuer":"","issueDate":""}}]}}

Rules:
- firstName / lastName: candidate's real name only, NEVER a city or job title
- skills: list every skill/technology mentioned as short strings
- workType: one of "REMOTE", "HYBRID", "ON_SITE", or "" if not mentioned
- currentlyWorking: true only if "present", "current", or "ongoing" appears for that job

Resume:
{text_chunk}"""

        response = model.generate_content(
            gemini_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192,
            )
        )
        if response and response.text:
            raw = response.text.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            open_count = raw.count('{')
            close_count = raw.count('}')
            if open_count > close_count:
                raw += '}' * (open_count - close_count)
            logger.info("Successfully parsed resume via Gemini (gemini-3-flash-preview).")
            return json.loads(raw)
    except Exception as e:
        logger.error(f"Gemini fallback parsing failed: {e}")


    return {}


class ResumeParser:
    """
    Clean, simple Groq-first resume parser.
    Call ResumeParser.parse(raw_text) → structured dict.
    """

    @staticmethod
    def parse(raw_text: str) -> Dict[str, Any]:
        # Clean up null bytes and control characters
        raw_text = raw_text.replace("\x00", "")
        raw_text = "".join(c for c in raw_text if c.isprintable() or c in "\n\r\t")
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

        # Extract email and phone via regex (fast, 100% reliable)
        email = _extract_email(raw_text)
        phone = _extract_phone(raw_text)

        # Primary extraction via Groq LLM
        data = _call_groq(raw_text)

        # ── personalDetails ───────────────────────────────────────────────
        pd = PersonalDetails(
            firstName=data.get("firstName") or "",
            lastName=data.get("lastName") or "",
            phone=phone or data.get("phone") or "",
            city=data.get("city") or "",
            province=data.get("province") or "",
            location=", ".join(filter(None, [
                data.get("city") or "",
                data.get("province") or "",
                data.get("country") or "",
            ])),
        )

        # ── skills ────────────────────────────────────────────────────────
        raw_skills = data.get("skills") or []
        skills = [
            Skill(name=s.strip(), level="Found")
            for s in raw_skills
            if isinstance(s, str) and s.strip()
        ]

        # ── education ────────────────────────────────────────────────────
        raw_edu = data.get("education") or []
        education = []
        for e in raw_edu:
            if not isinstance(e, dict):
                continue
            deg = e.get("degree") or ""
            if not deg:
                continue
            education.append(Education(
                degree=deg,
                institution=e.get("institution") or "Unknown Institution",
                fieldOfStudy=e.get("fieldOfStudy") or None,
                startDate=e.get("startDate") or None,
                endDate=e.get("endDate") or None,
            ))

        # ── work experience ───────────────────────────────────────────────
        raw_exp = data.get("workExperience") or []
        work_experience = []
        for w in raw_exp:
            if not isinstance(w, dict):
                continue
            if not (w.get("companyName") or w.get("role")):
                continue
            resp = w.get("responsibilities") or []
            work_experience.append(WorkExperience(
                companyName=w.get("companyName") or None,
                role=w.get("role") or None,
                jobTitle=w.get("role") or None,
                startDate=w.get("startDate") or None,
                endDate=w.get("endDate") or None,
                currentlyWorking=bool(w.get("currentlyWorking")),
                responsibilities=[r for r in resp if isinstance(r, str) and r.strip()],
            ))

        # ── projects ──────────────────────────────────────────────────────
        raw_proj = data.get("projects") or []
        projects = []
        for p in raw_proj:
            if not isinstance(p, dict) or not p.get("title"):
                continue
            tools = p.get("tools") or []
            projects.append(Project(
                title=p["title"],
                tools=[t for t in tools if isinstance(t, str) and t.strip()],
                startDate=p.get("startDate") or None,
                summary=p.get("summary") or None,
            ))

        # ── certifications ────────────────────────────────────────────────
        raw_certs = data.get("certifications") or []
        certifications = []
        for c in raw_certs:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            from app.models import Certificate
            certifications.append(Certificate(
                name=c["name"],
                issuer=c.get("issuer") or None,
                issueDate=c.get("issueDate") or None,
            ))

        return {
            "personalDetails": pd.model_dump(),
            "workType": data.get("workType") or None,
            "email": email,
            "skills": [s.model_dump() for s in skills],
            "education": [e.model_dump() for e in education],
            "workExperience": [w.model_dump() for w in work_experience],
            "projects": [p.model_dump() for p in projects],
            "certifications": [c.model_dump() for c in certifications],
            "summary": data.get("summary") or "",
        }


# Singleton for backward compatibility
parser = ResumeParser()
