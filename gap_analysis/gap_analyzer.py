
import re
from typing import List, Dict

from webargs import missing
from gap_analysis.llm_skill_extractor import extract_skills_with_llm
from similarity.similarity_service import cosine_similarity_np
from embeddings.embedding_service import generate_embedding

# ✅ Canonical skill vocabulary
SKILL_VOCABULARY = {
    "python",
    "django",
    "flask",
    "postgresql",
    "mysql",
    "mongodb",
    "sql",
    "rest api",
    "restful api",
    "api",
    "object oriented programming",
    "oop",
    "git",
    "html",
    "css",
    "javascript",
    "react",
    "web scraping",
    "data extraction",
    "data processing",
    "data analysis",
    "nlp",
    "natural language processing",
    "openai",
    "backend",
    "frontend",
    "software development",
    "software engineering",
    "cloud",
}


# ✅ Normalize skill variants
SKILL_SYNONYMS = {
    "apis": "api",
    "restful": "rest api",
    "restful apis": "rest api",
    "oop": "object oriented programming",
    "nlp": "natural language processing",
}

def _extract_candidate_skills(text: str) -> List[str]:
    """
    Naive skill extraction based on token frequency.
    This is an MVP approach and will be improved with LLMs later.
    """

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokens = text.split()

    # Very basic filtering
    stop_words = {
        "and", "or", "with", "in", "of", "for", "to", "on",
        "experience", "knowledge", "skills", "years"
    }

    skills = [
        token for token in tokens
        if len(token) > 2 and token not in stop_words
    ]

    # Unique skills (order preserved)
    return list(dict.fromkeys(skills))


def _clean_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_skills(text: str) -> List[str]:
    """
    Extract skills using vocabulary + n-gram matching.
    """
    text = _clean_text_for_matching(text)

    found_skills = set()

    # Check multi-word skills first
    for skill in SKILL_VOCABULARY:
        if " " in skill and skill in text:
            found_skills.add(skill)

    # Token-level check
    tokens = text.split()
    for token in tokens:
        normalized = SKILL_SYNONYMS.get(token, token)
        if normalized in SKILL_VOCABULARY:
            found_skills.add(normalized)

    return sorted(found_skills)

def simple_analyze_gaps(resume_text: str, jd_text: str) -> dict:
    """
    Analyze gaps between resume and job description.

    Returns:
        {
          "missing_skills": [...],
          "matching_skills": [...],
          "extra_skills": [...]
        }
    """
    
    
    if not resume_text or not jd_text:
        raise ValueError("Resume text and job description text are required")
    

    resume_skills = set(_extract_skills(resume_text))
    jd_skills = set(_extract_skills(jd_text))

    return {
        "matching_skills": sorted(resume_skills & jd_skills),
        "missing_skills": sorted(jd_skills - resume_skills),
        "extra_skills": sorted(resume_skills - jd_skills),
    }

    
    # matching_skills = sorted(resume_skills.intersection(jd_skills))
    # missing_skills = sorted(jd_skills.difference(resume_skills))
    # extra_skills = sorted(resume_skills.difference(jd_skills))
    
    
    # return {
    #     "matching_skills": matching_skills,
    #     "missing_skills": missing_skills,
    #     "extra_skills": extra_skills
    # }
  

def semantic_match(skill_a, skill_b, threshold=0.8):
    emb_a = generate_embedding(skill_a)
    emb_b = generate_embedding(skill_b)

    score = cosine_similarity_np(emb_a, emb_b)
    return score >= threshold
  

def canonicalize_skill(skill: str) -> str:
    skill = skill.lower().strip()

    replacements = {
        "framework": "",
        "development": "",
        "services": "",
        "tools": "",
        "platforms": "",
    }

    for k, v in replacements.items():
        skill = skill.replace(k, v)

    skill = " ".join(skill.split())
    return skill


def semantic_match_with_any(
    jd_skill: str,
    resume_skills: set,
    resume_skill_embedding,
    jd_skill_embedding,
    threshold: float = 0.8
) -> bool:
    jd_emb = jd_skill_embedding[jd_skill]

    for rs in resume_skills:
        if rs in jd_skill or jd_skill in rs:
            rs_emb = resume_skill_embedding[rs]
            if cosine_similarity_np(jd_emb, rs_emb) >= threshold:
                return True

    return False

def create_embeddings(skills: set[str]) -> dict[str, list[float]]:
    return {skill:generate_embedding(skill) for skill in skills}

def analyze_gaps_with_llm(
    resume_text: str,
    jd_text: str,
    llm_client
) -> Dict[str, List[str]]:
    """
    Perform gap analysis using LLM-extracted skills.
    """
    
    resume_skills_raw = set(extract_skills_with_llm(resume_text, llm_client))
    jd_skills_raw = set(extract_skills_with_llm(jd_text, llm_client))

    resume_skills = {canonicalize_skill(s) for s in resume_skills_raw}
    jd_skills = {canonicalize_skill(s) for s in jd_skills_raw}
    
    resume_skill_embedding = create_embeddings(resume_skills)
    jd_skill_embedding = create_embeddings(jd_skills)
    
    print('Embeddings created')

    matching=set()
    missing = set()
    
    # Exact matching 
    exact_matches = resume_skills & jd_skills
    matching |= exact_matches
    
    # Semantic fallback
    for jd_skill in jd_skills:
        if jd_skill not in matching:
            if semantic_match_with_any(
                jd_skill,
                resume_skills,
                resume_skill_embedding,
                jd_skill_embedding
            ):
                matching.add(jd_skill)
            else:
                missing.add(jd_skill)

    
    extra = resume_skills - matching

    return {
        "matching_skills": sorted(matching),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),
    }


    


