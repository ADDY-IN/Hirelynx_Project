import re
import logging
from typing import Optional
from .base import _llm_generate, scrape_website_text

logger = logging.getLogger(__name__)

async def summarize_employer_profile(employer_data: dict) -> str:
    """Generates a personalized company summary for the employer profile."""
    website = (employer_data.get("companyWebsite") or employer_data.get("website") or employer_data.get("websiteUrl") or employer_data.get("company_website") or "").strip()
    company_name = employer_data.get("companyName") or employer_data.get("legalName") or "The company"
    description = employer_data.get("companyDescription") or ""
    industry, company_type, company_size = employer_data.get("industry") or "", employer_data.get("companyType") or "", employer_data.get("companySize") or ""
    city, province, country = employer_data.get("city") or "", employer_data.get("province") or "", employer_data.get("country") or ""
    location_str = ", ".join([p for p in [city, province, country] if p])

    scraped_text = await scrape_website_text(website, timeout=30.0) if website else None
    has_rich_content = bool(scraped_text) or bool(description and len(description.strip()) > 30)

    if not has_rich_content and not website:
        raise ValueError("Please provide a Website URL or a Company Description to generate an AI summary.")

    known_facts = []
    if company_name and company_name != "The company": known_facts.append(f'Company name: "{company_name}"')
    if industry: known_facts.append(f"Industry: {industry}")
    if company_size: known_facts.append(f"Team size: {company_size} employees")
    if location_str: known_facts.append(f"Location: {location_str}")
    if description: known_facts.append(f'Employer description: "{description[:500]}"')

    facts_block = "\n".join(f"  • {f}" for f in known_facts) if known_facts else "  • (no additional data provided)"
    scraped_block = f"\n\nCONTENT SCRAPED FROM {website}:\n{scraped_text[:2000]}\n(TRUST THIS PRIMARY SOURCE)" if scraped_text else f"\n\n(Note: Website {website} provided but scraping failed.)" if website else ""

    prompt = f"""You are a world-class brand copywriter. Write a premium employer profile (6-8 sentences).
COMPANY DATA:
{facts_block}{scraped_block}
Rules:
- Write a strong hook about what they do.
- Sound human, third person.
- No buzzwords like "dynamic", "synergy", "cutting-edge".
- Use ONLY facts provided.
Return ONLY the profile text."""

    result = _llm_generate(prompt, max_tokens=800, temperature=0.9)
    if result:
        result = re.sub(r'\\(["\'/])', r'\1', result)
        result = re.sub(r'\\n', ' ', result)
        return " ".join(result.split())

    # Deterministic fallback — 5-6 sentence professional summary
    sentences = []

    # Junk Filter for description
    is_junk = False
    if description:
        desc_clean = description.strip()
        # Basic junk detection: no spaces or length < 10 or no vowels
        if " " not in desc_clean or len(desc_clean) < 15 or not re.search(r'[aeiouAEIOU]', desc_clean):
            is_junk = True

    # Sentence 1 — Identity hook
    identity_parts = [f"{company_name}"]
    if company_type:
        identity_parts.append(f"is a premier {company_type}")
    else:
        identity_parts.append("is a leading organization")
    
    if industry:
        identity_parts.append(f"shaping the future of the {industry} landscape")
    
    if location_str:
        identity_parts.append(f"with a strategic presence in {location_str}")
    sentences.append(" ".join(identity_parts) + ".")

    # Sentence 2 — Core focus or mission
    if description and not is_junk:
        desc_snippet = description.strip().rstrip(".")
        sentences.append(f"The organization is dedicated to {desc_snippet[:200]}, driving innovation through a commitment to excellence.")
    elif industry:
        sentences.append(f"Operating at the forefront of the {industry} sector, {company_name} is recognized for its ability to deliver high-impact solutions that meet the evolving needs of its global partners.")
    else:
        sentences.append(f"{company_name} is built on a foundation of operational excellence and a relentless pursuit of quality in everything they do.")

    # Sentence 3 — Team and Culture
    if company_size:
        sentences.append(f"With a talented team of {company_size} professionals, they have fostered a high-performance culture rooted in collaboration, agility, and shared success.")
    else:
        sentences.append(f"Their professional team is driven by a culture of agility and excellence, ensuring that every project reflects their commitment to superior results.")

    # Sentence 4 — Reach and Reputation
    if location_str:
        sentences.append(f"Headquartered in {location_str}, they have built a reputation for reliability and forward-thinking leadership within their domain.")
    else:
        sentences.append(f"They have established a distinguished reputation for reliability and forward-thinking leadership across all their operational channels.")

    # Sentence 5 — Growth and Hiring
    sentences.append(f"As they continue to expand their footprint, {company_name} is actively looking for exceptional talent to join their journey — and Hirelynx is proud to support their mission as their primary recruitment partner.")

    return " ".join(sentences)
