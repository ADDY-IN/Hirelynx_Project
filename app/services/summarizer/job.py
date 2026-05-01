import json
import logging
from typing import List, Optional
from .base import (
    _llm_generate, _EXP_LEVEL_LABELS, _EMPLOYMENT_LABELS, _WORK_SCHEDULE_LABELS
)

logger = logging.getLogger(__name__)

def generate_job_summary(job_data) -> str:
    """Generate a fully AI-written job post summary via Groq LLM."""
    def get(field: str, default=None):
        if isinstance(job_data, dict): return job_data.get(field, default)
        return getattr(job_data, field, default)

    city, province, country = (get("city") or "").strip(), (get("province") or "").strip(), (get("country") or "").strip()
    loc_parts = [p for p in [city, province, country] if p]
    location_str = ", ".join(loc_parts) if loc_parts else (get("location") or "").strip() or None

    sal_min, sal_max, currency = get("salaryMin"), get("salaryMax"), (get("currency") or "CAD").upper()
    salary_str = None
    if sal_min and sal_max: salary_str = f"{currency} {int(sal_min):,} – {int(sal_max):,}"
    elif sal_min: salary_str = f"{currency} {int(sal_min):,}+"

    exp_level, exp_min, exp_max = (get("experienceLevel") or "").replace("_", " ").title().strip() or None, get("experienceMin"), get("experienceMax")
    exp_str = exp_level
    if not exp_str and (exp_min is not None or exp_max is not None):
        if exp_min and exp_max: exp_str = f"{int(exp_min)}–{int(exp_max)} years"
        elif exp_min: exp_str = f"{int(exp_min)}+ years"

    job_dict = {
        k: v for k, v in {
            "jobTitle": (get("title") or "").strip(),
            "category": (get("category") or "").strip() or None,
            "employmentType": (get("employmentType") or "").replace("_", " ").title() or None,
            "experienceRequired": exp_str,
            "workSchedule": (get("workSchedule") or "").replace("_", " ").title() or None,
            "compensationType": (get("compensationType") or "").replace("_", " ").title() or None,
            "salaryRange": salary_str,
            "location": location_str,
            "remote": True if get("isRemote") else None,
            "responsibilities": (get("responsibilities") or [])[:6] or None,
            "requiredSkills": ((get("requiredSkills") or []) or (get("skills") or []))[:8] or None,
            "workAuthRequired": True if get("requiresWorkAuthorization") else None,
            "openToInternational": True if get("openToInternationalCandidates") else None,
            "description": ((get("description") or "")[:400]) or None,
        }.items() if v not in (None, [], "", False)
    }

    prompt = f"""You are writing a job posting for a job board. Write a compelling, professional summary paragraph for this role.
Rules:
- Write 6 to 8 complete sentences, under 230 words total
- Use third person ("We are looking for...", "The successful candidate...", "This role involves...")
- Sound like a real human wrote it — not a template or checklist
- End with a line about what makes this a good opportunity or team to join
- No bullet points — write flowing prose only
- Do NOT use: "dynamic", "synergy", "passionate", "cutting-edge", "innovative", "leverage", "empower"

Job details:
{json.dumps(job_dict, indent=2)}
Return ONLY the summary paragraph."""

    result = _llm_generate(prompt, max_tokens=600, temperature=0.75)
    if result: return result
    
    # --- Advanced Dynamic Fallback (No Groq Needed) ---
    title = job_dict.get("jobTitle", "professional")
    category = job_dict.get("category", "our organization")
    loc = job_dict.get("location")
    emp_type = job_dict.get("employmentType", "full-time").lower()
    exp = job_dict.get("experienceRequired")
    salary = job_dict.get("salaryRange")
    remote = job_dict.get("remote")
    skills = job_dict.get("requiredSkills", [])
    resps = job_dict.get("responsibilities", [])
    
    sentences = [f"We are actively seeking a highly dedicated {title} to join {category}."]
    sentences.append("As a core member of our team, you will be instrumental in executing high-level strategies and ensuring operational excellence.")
    
    if exp:
        sentences.append(f"This {emp_type} role requires {exp} of relevant experience to ensure you can make an immediate, measurable impact.")
    else:
        sentences.append(f"This is a {emp_type} position offering a highly collaborative, fast-paced, and rewarding environment.")
        
    if skills and len(skills) >= 2:
        sentences.append(f"Ideal applicants will bring strong proficiency in {skills[0]} and {skills[1]}, applying these capabilities to solve complex organizational challenges.")
    elif skills:
        sentences.append(f"Ideal applicants will bring strong proficiency in {skills[0]}, applying this core capability to elevate our standards.")
        
    if resps:
        r_text = str(resps[0]).lower().rstrip('.')
        sentences.append(f"Your day-to-day focus will primarily involve {r_text}, alongside other critical workflows.")
        
    sentences.append("The successful candidate will take ownership of key deliverables, drive core initiatives forward, and work closely with cross-functional leadership to achieve outstanding results.")
    
    if remote:
        sentences.append("This position offers the full flexibility of remote work, allowing you to contribute effectively from anywhere while maintaining a strong team connection.")
    elif loc:
        sentences.append(f"Based in {loc}, this opportunity provides an excellent platform to grow your career alongside a talented and driven group of professionals.")
    else:
        sentences.append("This opportunity provides an excellent platform to grow your career alongside a talented and driven group of professionals.")
        
    if salary:
        sentences.append(f"In addition to a competitive compensation package in the range of {salary}, we offer a culture that prioritizes continuous learning and long-term professional development.")
    else:
        sentences.append("We pride ourselves on offering a supportive culture that prioritizes continuous learning and long-term professional development.")
        
    return " ".join(sentences)
async def generate_personalized_responsibilities(job_title: str, noc_title: str, base_duties: List[str], company_name: Optional[str] = None, category: Optional[str] = None) -> List[str]:
    context = f"Job Title: {job_title}\nNOC Reference Title: {noc_title}"
    if company_name: context += f"\nCompany: {company_name}"
    if category: context += f"\nCategory: {category}"
    prompt = f"""You are an expert HR Consultant and Job Architect. Take standard NOC duties and rewrite them into 5-7 professional responsibilities.
{context}
Standard NOC Duties:
{chr(10).join(f"- {d}" for d in base_duties)}
Instructions:
- Use active, strong verbs.
- Tone must be premium and modern.
- Return ONLY a JSON array of strings."""
    result = _llm_generate(prompt)
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3].strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list): return parsed
        except Exception as e:
            logger.warning(f"Failed to parse AI responsibilities: {e}")
    return [d.strip() for d in base_duties[:6]]

async def generate_responsibilities_from_scratch(job_title: str, company_name: Optional[str] = None, category: Optional[str] = None) -> List[str]:
    context = f"Job Title: {job_title}"
    if company_name: context += f"\nCompany: {company_name}"
    if category: context += f"\nCategory: {category}"
    prompt = f"""You are an expert HR Consultant. Generate 5-7 professional responsibilities for this job title.
{context}
Instructions:
- Use active, strong verbs.
- Tone must be premium.
- Return ONLY a JSON array of strings."""
    result = _llm_generate(prompt)
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3].strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list): return parsed
        except Exception as e:
            logger.error(f"Failed to parse scratch responsibilities: {e}")
    return [f"Execute core functions related to the {job_title} role.", "Collaborate with team members to drive results."]
