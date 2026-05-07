import json
import re
from typing import List, Dict

SUGGESTION_PROMPT = """
You are an expert technical recruiter and resume reviewer.

Task:
Based on the information provided, generate actionable suggestions to improve the resume so it better matches the job description.

Rules:
- Use the gap analysis as the main source of truth.
- Focus on concrete, actionable improvements.
- Do NOT rewrite the resume.
- Do NOT invent experience.
- Be concise and professional.
- Return the output as a JSON array of strings.
- Do NOT include explanations outside the JSON.

Input:
Resume summary:
{resume_text}

Job description summary:
{job_description}

Similarity score:
{similarity_score}

Gap analysis:
Matching skills: {matching_skills}
Missing skills: {missing_skills}
Extra skills: {extra_skills}

Output format:
[
  "Suggestion 1",
  "Suggestion 2",
  "Suggestion 3"
]


Recent user feedback on previous suggestions is provided below.

Your task is to LEARN from this feedback and adjust your recommendations accordingly.

Guidelines for using feedback:
- Suggestions marked as POSITIVE indicate the style, level of detail, and focus that users prefer.
- Suggestions marked as NEGATIVE should be AVOIDED or REPHRASED.
- If negative feedback appears for suggestions that assume experience, use conditional language ("if applicable", "if you have experience").
- If negative feedback appears for generic suggestions, make recommendations more specific and tied to actual missing skills.
- Do NOT repeat suggestions that are similar to negatively rated ones.

Recent feedback:
Positive feedback:
{positive_feedback}

Negative feedback:
{negative_feedback}
"""


def generate_suggestions(
    resume_text: str,
    job_description: str,
    gap_analysis: Dict[str, List[str]],
    similarity_score: float,
    positive_feedback,
    negative_feedback,
    llm_client
) -> List[str]:
    prompt = SUGGESTION_PROMPT.format(
        resume_text=resume_text,  # safety limit
        job_description=job_description,
        similarity_score=round(similarity_score, 2),
        matching_skills=", ".join(gap_analysis["matching_skills"]),
        missing_skills=", ".join(gap_analysis["missing_skills"]),
        extra_skills=", ".join(gap_analysis["extra_skills"]),
        positive_feedback="\n- ".join(positive_feedback) if positive_feedback else "None",
        negative_feedback="\n- ".join(negative_feedback) if negative_feedback else "None"
    )

    raw_response = llm_client.generate(prompt)

    cleaned = re.sub(r"```json|```", "", raw_response).strip()

    try:
        suggestions = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON returned by LLM:\n{cleaned}"
        )

    return [s.strip() for s in suggestions if s.strip()]
