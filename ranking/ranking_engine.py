from typing import List
from ranking.candidate_result import CandidateResult
from preprocessing.text_cleaner import clean_text
from utils.save_uploaded_file import save_uploaded_file
from parsing.resume_parser import parse_resume
from similarity.similarity_service import cosine_similarity_np
from gap_analysis.gap_analyzer import analyze_gaps_with_llm
from embeddings.embedding_service import generate_embedding

def rank_candidates(results: List[CandidateResult]) -> List[CandidateResult]:
    """
    Rank candidates by similarity score (descending).
    """
    return sorted(
        results,
        key=lambda r: r.weighted_score,
        reverse=True
    )

def analyze_multiple_resumes(
    resume_files: list,
    clean_jd_text: str,
    critical_skills,
    llm_client
):
    results = []

    jd_embedding = generate_embedding(clean_jd_text)

    for resume_file in resume_files:
        resume_path = save_uploaded_file(resume_file)
        resume_data = parse_resume(resume_path)
        clean_resume = clean_text(resume_data["text"])

        resume_embedding = generate_embedding(clean_resume)
        score = cosine_similarity_np(resume_embedding, jd_embedding)

        gap = analyze_gaps_with_llm(
            resume_text=clean_resume,
            jd_text=clean_jd_text,
            llm_client=llm_client
        )
        
        weighted_score = compute_weighted_score(
            similarity_score=score,
            critical_skills=set(critical_skills),
            matching_skills=set(gap["matching_skills"]),
            missing_skills=set(gap["missing_skills"])
        )

        results.append(
            CandidateResult(
                candidate_id=resume_file.name,
                similarity_score=score,
                weighted_score=weighted_score,                matching_skills=gap["matching_skills"],
                missing_skills=gap["missing_skills"],
                extra_skills=gap["extra_skills"]
            )
        )

    return rank_candidates(results)


def compute_weighted_score(
    similarity_score: float,
    critical_skills: set,
    matching_skills: set,
    missing_skills: set
) -> float:

    matched_critical = critical_skills & matching_skills
    missing_critical = critical_skills & missing_skills

    score = similarity_score
    score += 0.05 * len(matched_critical)
    score -= 0.10 * len(missing_critical)

    return max(0.0, min(score, 1.0))