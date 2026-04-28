
from parsing.job_description_parser import parse_job_description
from parsing.resume_parser import parse_resume
from preprocessing.text_cleaner import clean_text
from embeddings.embedding_service import generate_embedding
from similarity.similarity_service import cosine_similarity_np
from gap_analysis.gap_analyzer import simple_analyze_gaps, analyze_gaps_with_llm
from llm.huggingface_client import HuggingFaceLLMClient

def main():
    print("Welcome to the AI Resume Analyzer!")
    print("This tool will help you analyze your resume against a job description.")
    print("Let's get started!")
    
    
    # --- Test paths ---
    resume_path = "data/raw/CV-Jessy Gigato.pdf"
    jd_path = "data/raw/job_description.docx"
    
    
    # --- Parsing ---
    print("Parsing resume...")
    resume_data = parse_resume(resume_path)
    print("Resume parsed successfully!", resume_data)
    
    print("Parsing job description...")
    jd_data = parse_job_description(file_path=jd_path)
    print("Job description parsed successfully!", jd_data)
    
    # --- Preprocessing ---
    print("Preprocessing texts...")
    clean_resume_text = clean_text(resume_data["text"])
    clean_jd_text = clean_text(jd_data["text"])

    # --- Embeddings ---
    print("Generating embeddings...")
    resume_embedding = generate_embedding(clean_resume_text)
    jd_embedding = generate_embedding(clean_jd_text)

    # --- Similarity ---
    print("Calculating similarity...")
    similarity_score = cosine_similarity_np(
        resume_embedding,
        jd_embedding
    )
    print(f"Similarity score: {similarity_score:.4f}")
    
    # --- Gap Analysis ---
    print("Running gap analysis...")
    llm_client = HuggingFaceLLMClient()
    gap_analysis = analyze_gaps_with_llm(
        resume_text=clean_resume_text,
        jd_text=clean_jd_text,
        llm_client=llm_client
    )
    # gap_analysis = simple_analyze_gaps(
    #     resume_text=clean_resume_text,
    #     jd_text=clean_jd_text
    # )

    print("\nGap analysis results:")
    print(f"Matching skills: {gap_analysis['matching_skills']}")
    print(f"Missing skills: {gap_analysis['missing_skills']}")
    print(f"Extra skills: {gap_analysis['extra_skills']}")


    
if __name__ == "__main__":
    main()