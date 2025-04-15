import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import parse_pdf, extract_resume_info, vectorize_text_glove

def recommend_jobs(resume_pdf_path, job_df, glove_embeddings, lambda_pref=0.75, top_k=3):
    # Extract and preprocess resume
    raw_text = parse_pdf(resume_pdf_path)
    resume_info = extract_resume_info("uploaded_resume", raw_text)
    resume_text = " ".join([resume_info.get(k, "") for k in ["title", "summary", "skills", "experience", "education"]])
    resume_vec = vectorize_text_glove(resume_text, glove_embeddings)

    # Ensure job_id and salary exist
    if 'job_id' not in job_df.columns:
        job_df['job_id'] = job_df.index.astype(str)
    if 'salary' not in job_df.columns:
        job_df['salary'] = job_df.get('max_salary', pd.Series(np.random.randint(30000, 150001, size=len(job_df))))

    # Vector and salary processing
    job_vectors = job_df[[col for col in job_df.columns if col.startswith("glove_")]].values
    job_salaries = job_df['salary'].values
    similarity_scores = cosine_similarity([resume_vec], job_vectors)[0]

    r_salary = resume_info.get("salary", 60000)  # fallback if not found
    salary_scores = 1 - np.abs(job_salaries - r_salary) / (150000 - 30000)
    salary_scores = np.clip(salary_scores, 0, 1)

    combined_scores = lambda_pref * similarity_scores + (1 - lambda_pref) * salary_scores
    top_idx = np.argsort(combined_scores)[::-1][:top_k]

    return job_df.iloc[top_idx].copy(), similarity_scores[top_idx], salary_scores[top_idx], combined_scores[top_idx]

