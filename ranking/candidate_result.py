from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CandidateResult:
    candidate_id: str
    similarity_score: float
    weighted_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]