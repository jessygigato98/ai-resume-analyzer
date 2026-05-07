import json
import re

CRITICAL_SKILLS_PROMPT = """
You are an expert technical recruiter.

Task:
From the job description below, identify the MOST CRITICAL technical skills
required to successfully perform this role.

Definition of critical skills:
- Skills that are essential for the role
- If a candidate lacks one of these, they would struggle significantly
- Typically core technologies, frameworks, or concepts

Rules:
- Extract ONLY technical skills (no soft skills).
- Limit the list to a maximum of 5 skills.
- Use canonical, concise skill names.
- Normalize skills to lowercase.
- Deduplicate semantically equivalent skills.
- Prefer core backend skills over secondary tools.
- Return ONLY valid JSON.
- Do NOT include explanations.

Output format:
{
  "critical_skills": ["skill1", "skill2", "..."]
}

Job Description:
"""

def extract_critical_skills(
    jd_text: str,
    llm_client
) -> list[str]:
    prompt = CRITICAL_SKILLS_PROMPT + jd_text

    raw_response = llm_client.generate(prompt)

    cleaned = re.sub(r"```json|```", "", raw_response).strip()

    try:
        data = json.loads(cleaned)
        skills = data.get("critical_skills", [])
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON returned by LLM for critical skills")

    return [s.strip().lower() for s in skills if s.strip()]