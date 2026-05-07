
import streamlit as st
# from streamlit.string_util import clean_text
from preprocessing.text_cleaner import clean_text
from embeddings.embedding_service import generate_embedding
import gap_analysis
from parsing.job_description_parser import parse_job_description
from parsing.resume_parser import parse_resume
from similarity.similarity_service import cosine_similarity_np
from gap_analysis.gap_analyzer import simple_analyze_gaps, analyze_gaps_with_llm
from llm.huggingface_client import HuggingFaceLLMClient
from suggestions.suggestion_service import generate_suggestions
from utils.save_feedback import save_feedback
from utils.feedback import load_recent_feedback,summarize_feedback
from utils.save_uploaded_file import save_uploaded_file
from ranking.ranking_engine import analyze_multiple_resumes
from ranking.critical_skills_extractor import extract_critical_skills

def render_skill_chips(skills, color="#E8F0FE", text_color="#1A73E8"):
    if not skills:
        st.write("—")
        return

    # 1. Usamos una lista para unir todo al final sin espacios extra
    chips = []
    for skill in skills:
        span = f'<span style="display:inline-block;padding:4px 12px;margin:4px;border-radius:16px;background-color:{color};color:{text_color};font-size:13px;font-weight:500;white-space:nowrap;">{skill}</span>'
        chips.append(span)

    full_html = f'<div style="display:flex;flex-wrap:wrap;">{"".join(chips)}</div>'
    
    st.markdown(full_html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    st.title("📄 AI Resume Analyzer")
    st.write(
        "Upload your resume and compare it against a job description. "
        "The system will analyze semantic similarity, identify gaps, "
        "and provide actionable improvement suggestions."
    )
    
    st.header("Analysis Mode")

    analysis_mode = st.radio(
        "Choose analysis type:",
        ["Single Resume Analysis", "Multiple Resume Ranking"]
    )

    if analysis_mode == "Single Resume Analysis":
        st.subheader("Single Resume vs Job Description")
        # --- Inputs ---
        st.header("1️⃣ Upload Inputs")
        st.subheader("Resume")
        resume_file = st.file_uploader(
            "Upload your resume (PDF or DOCX)",
            type=["pdf", "docx"]
        )
        
        st.markdown("---")
        
        # Job Description options
        st.subheader("Job Description")
        job_description_text = st.text_area(
            "Or paste the Job Description text",
            height=250,
            disabled=st.session_state.get("jd_file_uploaded", False)
        )
        jd_file = st.file_uploader(
            "Upload Job Description (PDF or DOCX) — optional",
            type=["pdf", "docx"],
            key="jd_file",
            disabled=bool(job_description_text.strip())
        )
        
        analyze_clicked = st.button("🚀 Analyze Resume")
        
        if analyze_clicked:
            if not resume_file or not (job_description_text.strip() or jd_file):
                st.error("Please upload a resume and provide a job description.")
                return
            
            with st.spinner("Running analysis..."):
                # --- Save resume ---
                resume_path = save_uploaded_file(resume_file)
                
                # --- Parse resume ---
                resume_data = parse_resume(resume_path)

                # --- Parse job description ---
                if jd_file is not None:
                    jd_path = save_uploaded_file(jd_file)
                    jd_data = parse_job_description(file_path=jd_path)
                else:
                    jd_data = parse_job_description(raw_text=job_description_text)

                # --- Preprocessing ---
                clean_resume_text = clean_text(resume_data["text"])
                clean_jd_text = clean_text(jd_data["text"])
                
                # --- Embeddings ---
                print("Generating embeddings...")
                resume_embedding = generate_embedding(clean_resume_text)
                jd_embedding = generate_embedding(clean_jd_text)

                
                # --- Similarity ---
                similarity_score = cosine_similarity_np(
                    resume_embedding,
                    jd_embedding
                )
                
                # --- Gap Analysis (LLM) ---
                llm_client = HuggingFaceLLMClient()
                gap_analysis = analyze_gaps_with_llm(
                    resume_text=clean_resume_text,
                    jd_text=clean_jd_text,
                    llm_client=llm_client
                )
                
                # --- Feedback ---
                recent_feedback = load_recent_feedback()
                feedback_summary = summarize_feedback(recent_feedback)
                positive_feedback = feedback_summary['positive']
                negative_feedback = feedback_summary['negative']
                
                # --- Suggestions ---
                suggestions = generate_suggestions(
                    resume_text=clean_resume_text,
                    job_description=clean_jd_text,
                    gap_analysis=gap_analysis,
                    similarity_score=similarity_score,
                    positive_feedback=positive_feedback,
                    negative_feedback=negative_feedback,
                    llm_client=llm_client
                )
                
                st.session_state["analysis_done"] = True
                st.session_state["similarity_score"] = similarity_score
                st.session_state["gap_analysis"] = gap_analysis
                st.session_state["suggestions"] = suggestions

            
        # --- Results ---
        if st.session_state.get("analysis_done"):
            st.header("2️⃣ Analysis Results")
            
            similarity_score = st.session_state["similarity_score"]
            gap_analysis = st.session_state["gap_analysis"]
            suggestions = st.session_state["suggestions"]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🔍 Similarity Score",
                    value=f"{similarity_score:.2f}"
                )
                st.subheader("✅ Matching Skills")
                render_skill_chips(gap_analysis["matching_skills"], color="#E6F4EA", text_color="#137333")
                
                st.subheader("❌ Missing Skills")
                render_skill_chips(gap_analysis["missing_skills"], color="#FCE8E6", text_color="#A50E0E")

                st.subheader("➕ Extra Skills")
                render_skill_chips(gap_analysis["extra_skills"], color="#FEF7E0", text_color="#B06000")
            
            with col2:
                st.subheader("💡 Resume Improvement Suggestions")
                for i, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"**{i}.** {suggestion}")
                    
                    col_yes, col_no = st.columns([1, 1])
                    with col_yes:
                        st.button(
                            "👍 Helpful",
                            key=f"yes_{i}",
                            on_click=save_feedback,
                            kwargs={
                                "suggestion":suggestion,
                                "rating":"positive",
                                "context":{
                                    "similarity_score": similarity_score,
                                    "gap_analysis": gap_analysis
                                }
                            })

                    with col_no:
                        st.button(
                            "👍 Not helpful",
                            key=f"no_{i}",
                            on_click=save_feedback,
                            kwargs={
                                "suggestion":suggestion,
                                "rating":"negative",
                                "context":{
                                    "similarity_score": similarity_score,
                                    "gap_analysis": gap_analysis
                                }
                            })
    elif analysis_mode == "Multiple Resume Ranking":
        st.subheader("Rank Multiple Resumes Against One Job Description")
        resume_files = st.file_uploader(
            "Upload multiple resumes (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="multi_resumes"
        )

        # Job Description options
        st.subheader("Job Description")
        job_description_text = st.text_area(
            "Or paste the Job Description text",
            height=250,
            disabled=st.session_state.get("jd_file_uploaded", False)
        )
        jd_file = st.file_uploader(
            "Upload Job Description (PDF or DOCX) — optional",
            type=["pdf", "docx"],
            key="jd_file",
            disabled=bool(job_description_text.strip())
        )

        rank_clicked = st.button("Rank Candidates", key="rank_analyze")
    
        if rank_clicked:
             # --- Parse job description ---
            if jd_file is not None:
                jd_path = save_uploaded_file(jd_file)
                jd_data = parse_job_description(file_path=jd_path)
            else:
                jd_data = parse_job_description(raw_text=job_description_text)
            
            # --- Preprocessing job description---
            clean_jd_text = clean_text(jd_data["text"])
            
            critical_skills = extract_critical_skills(
                jd_text=clean_jd_text,
                llm_client=HuggingFaceLLMClient()
            )

            ranked_results = analyze_multiple_resumes(
                resume_files=resume_files,
                clean_jd_text=clean_jd_text,
                critical_skills=critical_skills,
                llm_client=HuggingFaceLLMClient()
            )
            
            for i, candidate in enumerate(ranked_results, 1):
                st.subheader(f"#{i} – {candidate.candidate_id}")
                st.metric("Base Similarity", f"{candidate.similarity_score:.2f}")
                st.metric("Weighted Score", f"{candidate.weighted_score:.2f}")
                render_skill_chips(              
                    set(candidate.matching_skills) & set(critical_skills),
                        color="#E3F2FD",
                        text_color="#0D47A1"
                )




if __name__ == "__main__":
    main()