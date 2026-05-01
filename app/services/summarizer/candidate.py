import re
import json
import logging
from typing import List, Optional, Dict, Tuple
from .base import (
    _llm_generate, _infer_domains, _title_to_domain_str, 
    _infer_years, _pick, _DOMAIN_LABELS
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence pool generators — all work for ANY profession
# ---------------------------------------------------------------------------

def _opening(name: str, title: str, domain_str: str, years: Optional[int]) -> str:
    yr = f"over {years} year{'s' if years != 1 else ''}" if years else "several years of"
    t  = title or "professional"
    pools = [
        f"I am a dedicated {t} with {yr} experience in {domain_str}.",
        f"With {yr} experience in {domain_str}, I have built a strong track record for quality work and consistent results.",
        f"I bring {yr} hands-on expertise in {domain_str}, combining deep practical knowledge with a commitment to excellence.",
        f"As a skilled {t} with {yr} experience, I have developed a strong command of {domain_str}.",
        f"I am an accomplished {t} with a career spanning {yr} in {domain_str}.",
        f"Driven by a passion for my craft, I have spent {yr} honing my skills in {domain_str}.",
    ]
    return _pick(pools)

def _role_sentence(role: str, company: str, is_current: bool, responsibilities: List[str]) -> Tuple[str, bool]:
    r = role.strip().rstrip(".") if role else "a professional role"
    c = company.strip() if company else "a reputable organisation"
    highlight = ""
    if responsibilities:
        scored = sorted(responsibilities, key=lambda x: (bool(re.search(r"\d+", x)), len(x)), reverse=True)
        raw = scored[0][:130].strip().rstrip(".")
        highlight = raw[0].lower() + raw[1:] if raw else ""
    included = bool(highlight)
    if highlight:
        pools = [
            f"{'Currently serving' if is_current else 'Most recently'} as a {r} at {c}, I have {highlight}.",
            f"As a {r} at {c}, I have {highlight}, which has sharpened my practical expertise significantly.",
            f"At {c}, I serve as a {r} and have {highlight}.",
            f"In my role as {r} at {c}, I have {highlight}.",
        ]
    else:
        pools = [
            f"I currently hold the position of {r} at {c}." if is_current else f"Most recently, I worked as a {r} at {c}.",
            f"I have worked as a {r} at {c}.",
            f"Professionally, I serve as a {r} at {c}." if is_current else f"I have experience working as a {r} at {c}.",
        ]
    return _pick(pools), included

def _skills_sentence(skill_names: List[str], domain_str: str) -> str:
    top = skill_names[:6]
    if not top: return ""
    skill_list = ", ".join(top[:-1]) + f" and {top[-1]}" if len(top) > 1 else top[0]
    pools = [
        f"My toolkit includes {skill_list}, which I apply with precision across {domain_str}.",
        f"I am proficient in {skill_list} — skills that underpin my effectiveness in {domain_str}.",
        f"My core competencies span {skill_list}, making me a versatile contributor in {domain_str}.",
        f"I bring hands-on proficiency with {skill_list}, developed through real-world practice.",
        f"My skill set covers {skill_list} — capabilities honed through consistent, on-the-job application.",
    ]
    return _pick(pools)

def _education_sentence(degree: str, institution: str) -> str:
    d, s = degree.strip().rstrip("."), institution.strip()
    pools = [
        f"Academically, I hold a {d} from {s}, which gives me a strong foundation for my professional practice.",
        f"I completed a {d} at {s}, where I developed the theoretical grounding that informs my work.",
        f"My formal education includes a {d} from {s}.",
        f"Holding a {d} from {s}, I combine academic rigour with extensive practical experience.",
        f"I trained at {s}, graduating with a {d} that prepared me well for my career.",
    ]
    return _pick(pools)

def _achievement_sentence(responsibilities: List[str]) -> Optional[str]:
    quantified = [r for r in responsibilities if re.search(r"\d+\s*(?:%|k|x|hrs?|days?|weeks?|months?|years?|TB|GB|MB|users?|clients?|orders?|meals?|units?|projects?|accounts?|students?|patients?|employees?|staff|million|thousand|hundred|km|ft|sqft)", r, re.IGNORECASE)]
    if not quantified: return None
    raw = quantified[0][:130].strip().rstrip(".")
    h   = raw[0].lower() + raw[1:] if raw else ""
    pools = [f"Notably, I have {h}.", f"A highlight of my career is having {h}.", f"My impact is demonstrated by {h}."]
    return _pick(pools)

def _closing_sentence(domain_str: str, work_type: Optional[str]) -> str:
    wt = ""
    if work_type == "REMOTE": wt = ", and I am fully open to remote opportunities"
    elif work_type == "HYBRID": wt = ", and I am comfortable working in hybrid environments"
    pools = [
        f"I am deeply committed to continuous growth and delivering excellent results in {domain_str}{wt}.",
        f"Known for reliability and a strong work ethic, I bring real value to every team I join{wt}.",
        f"I thrive in collaborative, fast-paced settings and take pride in the quality of my work{wt}.",
        f"I am passionate about {domain_str} and always looking for opportunities to make a meaningful contribution{wt}.",
        f"Detail-oriented and highly motivated, I consistently go above and beyond in everything I take on{wt}.",
        f"Whether working independently or as part of a team, I approach every challenge with professionalism and dedication{wt}.",
    ]
    return _pick(pools)

# ---------------------------------------------------------------------------
# Main SummarizerService
# ---------------------------------------------------------------------------

class SummarizerService:
    def __init__(self):
        logger.info("Initializing Universal Candidate Summary Generator")

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Extractive summary for job descriptions."""
        if not text or len(text.strip()) < 50: return "Insufficient content to generate a summary."
        clean = " ".join(text.split())[:8000]
        sentences = re.split(r'\.\s+(?=[A-Z])|;\s+|\n', clean)
        def score(s: str) -> float:
            s = s.strip()
            if len(s) < 20: return 0.0
            sc = len(s) * 0.01
            if re.search(r'\b(lead|build|develop|design|manage|own|deliver|drive|scale|implement|prepare|serve|maintain|install|teach|advise|support|coordinate)\b', s, re.I): sc += 2
            if re.search(r'\d+', s): sc += 1.5
            if re.search(r'\b(experience|skills?|qualifications?|requirements?|responsibilities|duties)\b', s, re.I): sc += 1
            return sc
        ranked = sorted(sentences, key=score, reverse=True)
        top = [s.strip().rstrip(".") for s in ranked[:5] if s.strip() and len(s.strip()) > 20]
        if not top: return "Unable to generate a meaningful summary from the provided text."
        return ". ".join(top) + "."

    def summarize_candidate_profile(self, candidate) -> str:
        """Generates a natural first-person 'About Me' summary using Groq LLM."""
        from app.core.config import settings
        pd = candidate.personalDetails or {}
        if not isinstance(pd, dict): pd = {}
        skills = []
        for s in (candidate.skills or []):
            if isinstance(s, dict):
                name = s.get("name", "")
                if name: skills.append(name)
            elif isinstance(s, str) and s: skills.append(s)
        work_exp = []
        for w in (candidate.workExperience or []):
            if not isinstance(w, dict): continue
            work_exp.append({
                "role": w.get("jobTitle") or w.get("role") or "",
                "company": w.get("companyName") or "",
                "current": bool(w.get("currentlyWorking")),
                "responsibilities": (w.get("responsibilities") or [])[:3],
            })
        education = []
        for e in (candidate.education or []):
            if isinstance(e, dict):
                education.append({"degree": e.get("degree", ""), "institution": e.get("institution", "")})
        candidate_data = {
            "name": f"{pd.get('firstName','')} {pd.get('lastName','')}".strip() or None,
            "skills": skills[:10],
            "workExperience": work_exp,
            "education": education,
        }
        prompt = f"""Write a professional "About Me" summary for a candidate based on the data below.
Rules:
- Write in FIRST PERSON (I am..., I have..., I've...)
- 5-7 natural sentences, under 220 words
- No buzzwords like "passionate", "results-driven", "dynamic"
- Focus on REAL skills, experience, and impact
- Sound like a real human wrote it — not a robot template

Candidate data:
{json.dumps(candidate_data, indent=2)}
Return ONLY the summary text, nothing else."""
        result = _llm_generate(prompt)
        if result: return result
        
        # --- Advanced Dynamic Fallback (No Groq Needed) ---
        name = pd.get("firstName", "")
        role = work_exp[0]["role"] if work_exp else ""
        company = work_exp[0]["company"] if work_exp else ""
        is_current = work_exp[0]["current"] if work_exp else False
        resps = work_exp[0]["responsibilities"] if work_exp else []
        
        # Domain Inference
        domain_labels = _infer_domains(skills, role)
        domain_str = domain_labels[0] if domain_labels else _title_to_domain_str(role)
        years = _infer_years([w for w in (candidate.workExperience or []) if isinstance(w, dict)])
        
        # Build sentences
        sentences = []
        sentences.append(_opening(name, role, domain_str, years))
        
        role_sent, inc_resp = _role_sentence(role, company, is_current, resps)
        sentences.append(role_sent)
        
        if skills: sentences.append(_skills_sentence(skills, domain_str))
        if education: sentences.append(_education_sentence(education[0]["degree"], education[0]["institution"]))
        if not inc_resp and resps:
            ach = _achievement_sentence(resps)
            if ach: sentences.append(ach)
            
        sentences.append(_closing_sentence(domain_str, pd.get("workType", "")))
        
        return " ".join(sentences)

# Singleton for backward compatibility
summarizer_service = SummarizerService()
