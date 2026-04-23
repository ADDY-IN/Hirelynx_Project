"""
Universal Candidate Summary Generator
=======================================
Generates natural, AI-feeling first-person professional summaries for
candidates from ANY profession — tech, culinary, trades, healthcare,
finance, education, legal, creative, and beyond.

Design principles:
- Zero hardcoded profession assumptions in sentence templates
- Domain inference from TITLE first, then skills — falls back to the
  candidate's own title words, NEVER to a fixed category like "software engineering"
- Multi-pool sentence variation ensures every summary sounds unique
- Works for: chefs, plumbers, nurses, accountants, teachers, engineers,
  lawyers, designers, data scientists, warehouse workers, pilots — everyone
"""
import re
import json
import random
import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM helper — Groq for all summarization tasks
# ---------------------------------------------------------------------------
def _llm_generate(prompt: str) -> Optional[str]:
    """
    Generate text via Groq API. Returns None on failure so callers can fallback.
    """
    try:
        from app.config import settings
        from groq import Groq
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            return None
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq LLM unavailable: {e}")
    return None


# ---------------------------------------------------------------------------
# Universal profession domain clusters
# ---------------------------------------------------------------------------
_DOMAIN_CLUSTERS: Dict[str, List[str]] = {
    # ── Technology ──────────────────────────────────────────────────────────
    "software development": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust",
        "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
        "backend", "frontend", "full stack", "fullstack", "software engineer",
        "software developer", "programmer", "coding", "web development",
    ],
    "cloud & DevOps":       [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "devops", "cloud", "infrastructure", "sre", "linux",
        "helm", "prometheus", "grafana", "cloud engineer", "platform engineer",
    ],
    "data & analytics":     [
        "sql", "pandas", "spark", "hadoop", "dbt", "airflow", "bigquery", "snowflake",
        "tableau", "powerbi", "looker", "data analyst", "data engineer",
        "business analyst", "bi developer", "etl", "data warehouse",
    ],
    "AI & machine learning": [
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
        "scikit-learn", "bert", "gpt", "llm", "generative ai", "mlops",
        "computer vision", "data scientist", "ml engineer", "ai engineer",
        "huggingface", "langchain", "xgboost",
    ],
    "mobile development":   [
        "ios", "android", "flutter", "react native", "swift", "kotlin",
        "xcode", "mobile developer", "app developer", "jetpack compose",
    ],
    "cybersecurity":        [
        "cybersecurity", "penetration testing", "soc", "siem", "owasp",
        "fortify", "sonarqube", "compliance", "vulnerability", "firewall",
        "security analyst", "infosec", "ethical hacking", "network security",
    ],
    "product management":   [
        "product manager", "product owner", "roadmap", "agile", "scrum",
        "stakeholder", "kpi", "okr", "user story", "sprint", "backlog",
        "go-to-market", "product strategy", "ux", "user research",
    ],

    # ── Healthcare & Life Sciences ───────────────────────────────────────────
    "healthcare & nursing":  [
        "nurse", "nursing", "rn", "lpn", "patient care", "clinical", "icu",
        "er", "medication", "vital signs", "emr", "ehr", "healthcare",
        "hospital", "medical", "phlebotomy", "caregiver", "health aide",
    ],
    "medicine & pharmacy":   [
        "physician", "doctor", "md", "pharmacist", "pharmacy", "prescription",
        "diagnosis", "treatment", "surgery", "radiology", "lab technician",
        "physiotherapy", "occupational therapy", "dental", "dentist", "optometry",
    ],

    # ── Culinary & Hospitality ───────────────────────────────────────────────
    "culinary & food service": [
        "chef", "cook", "culinary", "kitchen", "food preparation", "menu",
        "catering", "pastry", "baking", "sous chef", "line cook",
        "food safety", "haccp", "restaurant", "barista", "bartender",
        "sommelier", "banquet", "food and beverage",
    ],
    "hospitality & tourism":   [
        "hotel", "hospitality", "front desk", "concierge", "housekeeping",
        "guest services", "travel", "tourism", "resort", "event coordinator",
        "reservation", "property management",
    ],

    # ── Skilled Trades ───────────────────────────────────────────────────────
    "skilled trades":          [
        "plumber", "plumbing", "electrician", "electrical", "carpenter",
        "carpentry", "welder", "welding", "hvac", "mechanic", "technician",
        "pipefitter", "millwright", "ironworker", "sheet metal", "mason",
        "tiler", "roofer", "painter", "drywaller", "gas fitter",
        "red seal", "apprentice", "journeyman",
    ],
    "construction":            [
        "construction", "site supervisor", "project manager", "general contractor",
        "foreman", "estimator", "blueprint", "autocad", "scheduling",
        "site management", "project coordination", "building", "renovation",
    ],

    # ── Finance & Accounting ─────────────────────────────────────────────────
    "finance & accounting":    [
        "accountant", "accounting", "cpa", "bookkeeper", "bookkeeping",
        "tax", "audit", "financial analyst", "cfo", "controller",
        "accounts payable", "accounts receivable", "payroll", "quickbooks",
        "xero", "sage", "gaap", "ifrs", "financial reporting", "budget",
        "forecasting",
    ],
    "banking & investment":    [
        "banking", "bank", "investment", "portfolio", "wealth management",
        "equity", "fixed income", "risk management", "trading", "broker",
        "financial advisor", "mutual fund", "compliance", "aml",
        "capital markets",
    ],

    # ── Education & Training ─────────────────────────────────────────────────
    "education & teaching":    [
        "teacher", "teaching", "educator", "instructor", "professor",
        "curriculum", "lesson plan", "classroom", "student", "school",
        "academic", "tutor", "e-learning", "training", "workshop",
        "educational technology",
    ],

    # ── Legal ────────────────────────────────────────────────────────────────
    "legal":                   [
        "lawyer", "attorney", "solicitor", "barrister", "paralegal",
        "legal assistant", "contract", "litigation", "compliance",
        "corporate law", "real estate law", "immigration", "intellectual property",
        "dispute resolution", "arbitration",
    ],

    # ── Marketing & Sales ────────────────────────────────────────────────────
    "marketing & growth":      [
        "marketing", "digital marketing", "seo", "sem", "social media",
        "content marketing", "email marketing", "brand", "advertising",
        "campaign", "google ads", "meta ads", "crm", "hubspot",
        "salesforce", "lead generation",
    ],
    "sales & business dev":    [
        "sales", "account executive", "business development", "bdr", "sdr",
        "closing", "pipeline", "quota", "revenue", "b2b", "b2c",
        "client relations", "customer success", "territory management",
    ],

    # ── Creative & Design ────────────────────────────────────────────────────
    "design & creative":       [
        "graphic designer", "ui designer", "ux designer", "product designer",
        "illustrator", "figma", "sketch", "adobe", "photoshop", "indesign",
        "motion graphics", "video editing", "photography", "brand design",
        "typography",
    ],

    # ── Logistics & Supply Chain ─────────────────────────────────────────────
    "logistics & operations":  [
        "logistics", "supply chain", "warehouse", "inventory", "forklift",
        "shipping", "receiving", "dispatch", "fleet management",
        "procurement", "vendor management", "operations manager",
        "lean", "six sigma",
    ],

    # ── Human Resources ──────────────────────────────────────────────────────
    "human resources":         [
        "hr", "human resources", "recruiting", "recruiter", "talent acquisition",
        "onboarding", "offboarding", "hris", "workday", "bamboohr",
        "compensation", "benefits", "employee relations", "labour relations",
        "performance management",
    ],
}

# Humanised labels for each domain — used in sentence construction
_DOMAIN_LABELS: Dict[str, str] = {k: k for k in _DOMAIN_CLUSTERS}


def _infer_domains(skill_names: List[str], job_title: str) -> List[str]:
    """
    Infer the top 1–2 professional domains from skills and job title.
    Uses word-boundary regex to avoid false positives (e.g. 'er' from
    'healthcare' matching inside 'Water Supply Systems').
    Job title is included so a 'Head Chef' with no skills still maps correctly.
    """
    # Combine all terms into a single searchable corpus
    combined_terms = [s.lower() for s in skill_names] + [job_title.lower()]

    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_CLUSTERS.items():
        score = 0
        for kw in keywords:
            # Use word-boundary search for single tokens; phrase match for multi-word keywords
            if " " in kw:
                # Multi-word phrase: match as exact phrase within any term
                pattern = re.escape(kw)
            else:
                # Single word: require word boundaries to prevent 'er' matching 'water'
                pattern = rf"\b{re.escape(kw)}\b"
            for term in combined_terms:
                if re.search(pattern, term):
                    score += 1
                    break   # count each keyword at most once
        if score > 0:
            scores[domain] = score

    return sorted(scores, key=lambda k: scores[k], reverse=True)[:2]


def _title_to_domain_str(job_title: str) -> str:
    """
    Derive a human-friendly domain phrase directly from the job title
    when no cluster match was found. E.g. 'Master Electrician' → 'the
    electrical trades', 'Executive Chef' → 'the culinary arts'.
    """
    t = job_title.lower().strip()
    # Generic fallback mappings by title keyword
    fallbacks = [
        (r"\b(chef|cook|pastry|baker|culinary)\b",   "the culinary arts"),
        (r"\b(plumb|hvac|pipefitter|gas fitter)\b",  "plumbing and mechanical systems"),
        (r"\b(electri|wiring|electrician)\b",         "electrical systems and installation"),
        (r"\b(nurse|nursing|rn|lpn)\b",               "healthcare and patient care"),
        (r"\b(doctor|physician|surgeon|md)\b",         "medicine and clinical practice"),
        (r"\b(teacher|instructor|educator|tutor)\b",  "education and curriculum delivery"),
        (r"\b(lawyer|attorney|paralegal|legal)\b",    "legal practice and advisory"),
        (r"\b(accountant|cpa|bookkeep|auditor)\b",    "accounting and financial reporting"),
        (r"\b(designer|ux|ui|creative|illustrator)\b","design and creative direction"),
        (r"\b(sales|account exec|bdr|sdr)\b",         "sales and business development"),
        (r"\b(logistics|warehouse|supply chain)\b",   "logistics and supply chain management"),
        (r"\b(hr|recruiter|talent|human resource)\b", "human resources and people operations"),
        (r"\b(market|seo|brand|advertising)\b",       "marketing and brand growth"),
        (r"\b(construct|foreman|site super|estimat)\b","construction and project delivery"),
        (r"\b(carpenter|welder|mason|roofer)\b",      "skilled trades and craftsmanship"),
        (r"\b(mechanic|technician|auto)\b",           "automotive and mechanical services"),
    ]
    for pattern, label in fallbacks:
        if re.search(pattern, t):
            return label
    # Last resort: use the title itself
    return f"the {job_title.strip().lower()} field" if job_title else "their professional field"


def _infer_years(work_experience: List[Dict]) -> Optional[int]:
    """Estimate total years of experience from date ranges."""
    total = 0
    for exp in work_experience:
        start = exp.get("startDate", "")
        end   = exp.get("endDate", "")
        curr  = exp.get("currentlyWorking", False)
        sy = re.search(r"\b(20\d{2}|19\d{2})\b", str(start))
        ey = re.search(r"\b(20\d{2}|19\d{2})\b", str(end))
        if sy:
            s_yr = int(sy.group(1))
            e_yr = 2026 if curr else (int(ey.group(1)) if ey else s_yr + 1)
            total += max(0, e_yr - s_yr)
    return total if total > 0 else None


def _pick(pool: List[str]) -> str:
    return random.choice(pool)


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


def _role_sentence(role: str, company: str, is_current: bool,
                   responsibilities: List[str]) -> Tuple[str, bool]:
    """Returns (sentence, highlight_included)."""
    r = role.strip().rstrip(".") if role else "a professional role"
    c = company.strip() if company else "a reputable organisation"

    highlight = ""
    if responsibilities:
        scored = sorted(
            responsibilities,
            key=lambda x: (bool(re.search(r"\d+", x)), len(x)),
            reverse=True,
        )
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
    if not top:
        return ""
    skill_list = (
        ", ".join(top[:-1]) + f" and {top[-1]}" if len(top) > 1 else top[0]
    )
    pools = [
        f"My toolkit includes {skill_list}, which I apply with precision across {domain_str}.",
        f"I am proficient in {skill_list} — skills that underpin my effectiveness in {domain_str}.",
        f"My core competencies span {skill_list}, making me a versatile contributor in {domain_str}.",
        f"I bring hands-on proficiency with {skill_list}, developed through real-world practice.",
        f"My skill set covers {skill_list} — capabilities honed through consistent, on-the-job application.",
    ]
    return _pick(pools)


def _education_sentence(degree: str, institution: str) -> str:
    d = degree.strip().rstrip(".")
    s = institution.strip()
    pools = [
        f"Academically, I hold a {d} from {s}, which gives me a strong foundation for my professional practice.",
        f"I completed a {d} at {s}, where I developed the theoretical grounding that informs my work.",
        f"My formal education includes a {d} from {s}.",
        f"Holding a {d} from {s}, I combine academic rigour with extensive practical experience.",
        f"I trained at {s}, graduating with a {d} that prepared me well for my career.",
    ]
    return _pick(pools)


def _achievement_sentence(responsibilities: List[str]) -> Optional[str]:
    quantified = [
        r for r in responsibilities
        if re.search(
            r"\d+\s*(?:%|k|x|hrs?|days?|weeks?|months?|years?|TB|GB|MB|users?|"
            r"clients?|orders?|meals?|units?|projects?|accounts?|students?|"
            r"patients?|employees?|staff|million|thousand|hundred|km|ft|sqft)",
            r, re.IGNORECASE
        )
    ]
    if not quantified:
        return None
    raw = quantified[0][:130].strip().rstrip(".")
    h   = raw[0].lower() + raw[1:] if raw else ""
    pools = [
        f"Notably, I have {h}.",
        f"A highlight of my career is having {h}.",
        f"My impact is demonstrated by {h}.",
    ]
    return _pick(pools)


def _closing_sentence(domain_str: str, work_type: Optional[str]) -> str:
    wt = ""
    if work_type == "REMOTE":
        wt = ", and I am fully open to remote opportunities"
    elif work_type == "HYBRID":
        wt = ", and I am comfortable working in hybrid environments"
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
# Job Description (JD) Summary Generator
# ---------------------------------------------------------------------------

# Human-readable display maps for enum-style fields
_EXP_LEVEL_LABELS: Dict[str, str] = {
    "ENTRY":           "entry-level candidates",
    "JUNIOR":          "junior professionals (0–2 years)",
    "ONE_TO_THREE":    "candidates with 1–3 years of experience",
    "TWO_TO_FIVE":     "mid-level professionals (2–5 years)",
    "THREE_TO_FIVE":   "professionals with 3–5 years of experience",
    "FIVE_TO_TEN":     "seasoned professionals with 5–10 years of experience",
    "MID":             "mid-level professionals",
    "SENIOR":          "senior professionals with 5+ years of experience",
    "LEAD":            "lead-level professionals",
    "MANAGER":         "managers and team leads",
    "DIRECTOR":        "director-level professionals",
    "EXECUTIVE":       "executive-level leaders",
    "OVER_TEN":        "highly experienced professionals with 10+ years",
}

_EMPLOYMENT_LABELS: Dict[str, str] = {
    "FULL_TIME":   "full-time",
    "PART_TIME":   "part-time",
    "CONTRACT":    "contract",
    "INTERNSHIP":  "internship",
    "TEMPORARY":   "temporary",
    "FREELANCE":   "freelance",
}

_WORK_SCHEDULE_LABELS: Dict[str, str] = {
    "DAY":       "day shift",
    "EVENING":   "evening shift",
    "NIGHT":     "night shift",
    "ROTATING":  "rotating shifts",
    "FLEXIBLE":  "flexible hours",
    "WEEKENDS":  "weekends",
}


def generate_job_summary(job_data) -> str:
    """
    Generate a job post summary using local Ollama (llama3.1).
    If Ollama is not running, falls back to a simple template sentence.
    """
    def get(field: str, default=None):
        if isinstance(job_data, dict):
            return job_data.get(field, default)
        return getattr(job_data, field, default)

    # Build compact job dict for the prompt
    job_dict = {
        "title":            (get("title") or "").strip(),
        "location":         (get("location") or "").strip() or None,
        "employmentType":   (get("employmentType") or "").replace("_", " ").title() or None,
        "experienceMin":    get("experienceMin"),
        "experienceMax":    get("experienceMax"),
        "salaryMin":        get("salaryMin"),
        "salaryMax":        get("salaryMax"),
        "currency":         (get("currency") or "CAD").upper(),
        "isRemote":         get("isRemote") or False,
        "responsibilities": (get("responsibilities") or [])[:5],
        "requiredSkills":   (get("requiredSkills") or [])[:6],
        "description":      ((get("description") or "")[:300]) or None,
    }
    job_dict = {k: v for k, v in job_dict.items() if v not in (None, [], "")}

    prompt = f"""Write a professional job post summary paragraph for this job.

Rules:
- 6-8 natural sentences, under 220 words
- Written in third person (e.g. "We are looking for...", "The ideal candidate...")
- Sound like a real job posting, not a template
- Include the most important details (role, skills, location, experience if present)
- No buzzwords like "dynamic", "synergy", "passionate"

Job data:
{json.dumps(job_dict, indent=2)}

Return ONLY the summary paragraph, nothing else."""

    result = _llm_generate(prompt)
    if result:
        return result

    # Simple fallback if Ollama is not running
    title = job_dict.get("title", "professional")
    skills = job_dict.get("requiredSkills", [])
    location = job_dict.get("location", "")
    sk_str = ", ".join(skills[:4]) if skills else ""
    loc_str = f" based in {location}" if location else ""
    return (
        f"We are looking for a {title}{loc_str}."
        + (f" The ideal candidate will have experience with {sk_str}." if sk_str else "")
        + " This is an exciting opportunity to join a growing team."
    )


# ---------------------------------------------------------------------------
# Employer Website Scraper
# ---------------------------------------------------------------------------

async def scrape_website_text(url: str, timeout: float = 15.0) -> Optional[str]:
    """
    Fetches and extracts meaningful text from a company website.
    Tries homepage, then /about and /about-us sub-pages.
    Returns None on any failure.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; HirelynxBot/1.0; +https://hirelynx.com)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }

        def _extract_text(html: str) -> str:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "svg", "img"]):
                tag.decompose()
            for section_id in ("about", "mission", "company", "overview", "who-we-are"):
                el = soup.find(id=section_id) or soup.find(class_=section_id)
                if el:
                    text = el.get_text(separator=" ", strip=True)
                    if len(text) > 100:
                        return text[:2500]
            body = soup.find("body")
            return (body or soup).get_text(separator=" ", strip=True)[:2500]

        extracted_parts: list[str] = []

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    text = _extract_text(resp.text)
                    if text:
                        extracted_parts.append(text)
            except Exception:
                pass

            if len(" ".join(extracted_parts)) < 300:
                base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
                for path in ["/about", "/about-us", "/company"]:
                    try:
                        resp = await client.get(urljoin(base, path), headers=headers)
                        if resp.status_code == 200:
                            text = _extract_text(resp.text)
                            if text:
                                extracted_parts.append(text)
                                break
                    except Exception:
                        continue

        combined = " ".join(extracted_parts)
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined[:2500] if combined else None

    except Exception as e:
        logger.warning(f"Website scrape failed for {url}: {e}")
        return None


async def summarize_employer_profile(employer_data: dict) -> str:
    """
    Generates a personalized company summary for the employer profile.

    Flow:
      1. Scrape company website (up to 15s)
      2. Merge scraped text + employer data into a rich Groq prompt
      3. Dedicated Groq call (higher temp + more tokens than shared helper)
         so every company gets a genuinely unique, AI-written summary
      4. Fallback to a minimal template — raw description NEVER included
    """
    company_name = employer_data.get("companyName", "The company")
    description  = employer_data.get("companyDescription") or ""
    industry     = employer_data.get("industry") or ""
    company_type = employer_data.get("companyType") or ""
    company_size = employer_data.get("companySize") or ""
    city         = employer_data.get("city") or ""
    province     = employer_data.get("province") or ""
    country      = employer_data.get("country") or ""
    website      = employer_data.get("companyWebsite") or ""
    legal_name   = employer_data.get("legalName") or ""

    location_parts = [p for p in [city, province, country] if p]
    location_str   = ", ".join(location_parts)

    # --- Step 1: Scrape website ---
    scraped_text: Optional[str] = None
    if website:
        scraped_text = await scrape_website_text(website, timeout=15.0)
        if scraped_text:
            logger.info(f"Scraped {len(scraped_text)} chars from {website}")
        else:
            logger.info(f"Scrape returned no content for {website}")

    # --- Step 2: Build rich, data-anchored prompt ---
    # List only the fields that actually have content so the LLM can't
    # hide behind "not specified" filler and must reference real details.
    known_facts: list[str] = []
    if company_name and company_name != "The company":
        known_facts.append(f'Company name: "{company_name}"')
    if legal_name and legal_name != company_name:
        known_facts.append(f'Legal / registered name: "{legal_name}"')
    if industry:
        known_facts.append(f"Industry: {industry}")
    if company_type:
        known_facts.append(f"Organisation type: {company_type}")
    if company_size:
        known_facts.append(f"Team size: {company_size} employees")
    if location_str:
        known_facts.append(f"Location: {location_str}")
    if website:
        known_facts.append(f"Website: {website}")
    if description:
        known_facts.append(f'Employer description: "{description[:500]}"')

    facts_block = "\n".join(f"  • {f}" for f in known_facts) if known_facts else "  • (no additional data provided)"

    scraped_block = ""
    if scraped_text:
        scraped_block = (
            f"\n\nCONTENT SCRAPED FROM {website}:\n"
            f"{scraped_text[:1500]}\n"
            "(Use this as the primary source of truth about the company.)"
        )

    prompt = f"""You are a senior copywriter writing employer profiles for a professional job board.

COMPANY DATA:
{facts_block}{scraped_block}

YOUR TASK:
Write a 4–5 sentence company profile for this employer's job board page.

STRICT REQUIREMENTS — read carefully:
1. Every sentence MUST reference at least one specific, concrete detail from the data above
   (company name, industry, location, team size, what they actually do, etc.).
2. NO two summaries you write should ever sound the same — vary your sentence openings and structure.
3. Third-person voice only (e.g. "{company_name} is...", "The team at {company_name}...", "Based in {location_str or 'their region'}...").
4. Sound like a human copywriter wrote it — warm, professional, factual.
5. Do NOT use: "dynamic", "synergy", "passionate", "cutting-edge", "innovative solutions", "world-class".
6. Do NOT copy the employer description verbatim.
7. Do NOT invent facts that are not in the data.
8. If scraped website content is provided, derive your key facts from it — it is more reliable than the employer description.
9. Output ONLY the summary paragraph. No headings, no bullet points, no extra commentary.

Write the summary now:"""

    # --- Step 3: Dedicated Groq call — higher temp + more tokens than shared helper ---
    try:
        from app.config import settings
        from groq import Groq

        api_key = getattr(settings, "GROQ_API_KEY", None)
        if api_key:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,   # higher than shared helper → more creative variance
                max_tokens=550,     # room for 4-5 rich sentences
            )
            result = resp.choices[0].message.content.strip()
            # Scrape content can contain raw escaped quotes that the LLM echoes
            # literally — normalise them before returning.
            result = result.replace('\\"', '"').replace("\\n", " ")
            result = " ".join(result.split())   # collapse any stray whitespace
            if result:
                return result
    except Exception as e:
        logger.warning(f"Groq employer summary failed: {e}")

    # --- Step 4: Minimal clean fallback — no raw description ever ---
    parts: list[str] = [f"{company_name} is"]
    if company_type:
        parts.append(f"a {company_type}")
    if industry:
        parts.append(f"operating in the {industry} sector")
    if location_str:
        parts.append(f"based in {location_str}")
    base = " ".join(parts).rstrip() + "."
    if company_size:
        base += f" With a team of {company_size} employees, they are actively growing and hiring on Hirelynx."
    return base








# ---------------------------------------------------------------------------
# Main SummarizerService
# ---------------------------------------------------------------------------

class SummarizerService:

    def __init__(self):
        logger.info("Initializing Universal Candidate Summary Generator")

    # ------------------------------------------------------------------
    # JD extractive summarizer
    # ------------------------------------------------------------------
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Extractive summary for job descriptions — used by JD summarize endpoint."""
        if not text or len(text.strip()) < 50:
            return "Insufficient content to generate a summary."

        clean = " ".join(text.split())[:8000]
        sentences = re.split(r'\.\s+(?=[A-Z])|;\s+|\n', clean)

        def score(s: str) -> float:
            s = s.strip()
            if len(s) < 20:
                return 0.0
            sc = len(s) * 0.01
            if re.search(r'\b(lead|build|develop|design|manage|own|deliver|drive|scale|implement|prepare|serve|maintain|install|teach|advise|support|coordinate)\b', s, re.I):
                sc += 2
            if re.search(r'\d+', s):
                sc += 1.5
            if re.search(r'\b(experience|skills?|qualifications?|requirements?|responsibilities|duties)\b', s, re.I):
                sc += 1
            return sc

        ranked = sorted(sentences, key=score, reverse=True)
        top = [s.strip().rstrip(".") for s in ranked[:5] if s.strip() and len(s.strip()) > 20]
        if not top:
            return "Unable to generate a meaningful summary from the provided text."
        return ". ".join(top) + "."

    # ------------------------------------------------------------------
    # Candidate bio generator — works for ANY profession
    # ------------------------------------------------------------------
    def summarize_candidate_profile(self, candidate) -> str:
        """
        Generates a natural first-person 'About Me' summary using Groq LLM.
        Falls back to a simple sentence if Groq is unavailable.
        """
        import json as json_mod
        from app.config import settings

        # Build a compact candidate dict to send to the LLM
        pd = candidate.personalDetails or {}
        if not isinstance(pd, dict):
            pd = {}

        skills = []
        for s in (candidate.skills or []):
            if isinstance(s, dict):
                name = s.get("name", "")
                if name:
                    skills.append(name)
            elif isinstance(s, str) and s:
                skills.append(s)

        work_exp = []
        for w in (candidate.workExperience or []):
            if not isinstance(w, dict):
                continue
            work_exp.append({
                "role": w.get("jobTitle") or w.get("role") or "",
                "company": w.get("companyName") or "",
                "current": bool(w.get("currentlyWorking")),
                "responsibilities": (w.get("responsibilities") or [])[:3],
            })

        education = []
        for e in (candidate.education or []):
            if isinstance(e, dict):
                education.append({
                    "degree": e.get("degree", ""),
                    "institution": e.get("institution", ""),
                })

        candidate_data = {
            "name": f"{pd.get('firstName','')} {pd.get('lastName','')}".strip() or None,
            "skills": skills[:10],
            "workExperience": work_exp,
            "education": education,
        }

        api_key_present = bool(getattr(settings, "GROQ_API_KEY", None))

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

        # Try LLM: Groq (production) → Ollama (local dev)
        result = _llm_generate(prompt)
        if result:
            return result

        # Fallback: simple constructed sentence
        role = work_exp[0]["role"] if work_exp else "professional"
        sk = ", ".join(skills[:4]) if skills else ""
        return f"I am a {role}{' with skills in ' + sk if sk else ''}. I bring hands-on experience and a strong work ethic to every role I take on."




# Singleton
summarizer_service = SummarizerService()
