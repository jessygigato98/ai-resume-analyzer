
import json
import re
from typing import List


SKILL_EXTRACTION_PROMPT = """
You are an information extraction system.

Task:
Extract a single list of professional skills from the text below.
The list must include both technical skills and relevant soft skills.

IMPORTANT:
When multiple phrases refer to the SAME underlying skill, you MUST return
ONLY ONE canonical skill name.

Examples of canonicalization (technical):
- "restful api consumption", "restful api development", "rest api usage" → "rest api"
- "django framework", "django web framework" → "django"
- "flask framework" → "flask"
- "object-oriented programming principles" → "object-oriented programming"
- "nosql databases", "mongodb database" → "mongodb"


Canonicalization examples (soft skills):
- "team collaboration", "collaborating with cross-functional teams" → "team collaboration"
- "problem-solving skills", "strong problem-solving ability" → "problem solving"
- "good communication skills", "effective communication" → "communication"
- "analytical thinking", "analytical skills" → "analytical thinking"

Rules:
- Extract ONLY professional skills (technical or soft skills).
- Technical skills include: programming languages, frameworks, tools, databases, APIs, platforms.
- Soft skills include: problem solving, communication, teamwork, analytical thinking, leadership, adaptability.
- Do NOT include personality traits (e.g., "motivated", "hardworking", "passionate").
- Do NOT include years, dates, job titles, or responsibilities.
- Normalize skills to lowercase.
- Use the MOST COMMON and SHORTEST canonical name
- Remove qualifiers such as: development, consumption, usage, principles, tools, services, platforms.
- Deduplicate semantically equivalent skills
- Limit soft skills to the MOST RELEVANT ones (max 8–10).
- Return ONLY valid JSON.
- Do NOT add explanations.

Output format:
{
  "skills": ["skill1", "skill2", "..."]
}

Text:
"""

def extract_skills_with_llm(
    text: str,
    llm_client
) -> List[str]:
    """
    Extract skills from text using an LLM.
    """

    prompt = SKILL_EXTRACTION_PROMPT + text

    response = llm_client.generate(prompt)

    # --- Safety cleanup ---
    response = response.strip()
    response = re.sub(r"```json|```", "", response)

    try:
        data = json.loads(response)
        skills = data.get("skills", [])
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON")

    # Normalize + deduplicate
    skills = sorted(set(skill.strip().lower() for skill in skills if skill.strip()))

    return skills