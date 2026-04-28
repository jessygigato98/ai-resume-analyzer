
import json
import re
from typing import List


SKILL_EXTRACTION_PROMPT = """
You are an information extraction system.

Task:
Extract a list of professional skills from the text below.

Rules:
- Extract only skills (technical or professional skills).
- Do NOT include soft skills unless explicitly technical.
- Do NOT include years, dates, job titles, or responsibilities.
- Normalize skills to lowercase.
- Use concise skill names.
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