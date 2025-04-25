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

# --- Recommendation function using vectorized numpy operations ---
def recommend_jobs_2(resume_pdf_path, job_df, glove_embeddings, FILTERED_VECTORS, FILTERED_NORMS, FILTERED_SALARIES, FILTERED_INDEX, lambda_pref=0.75, top_k=3):
    # 1. Extract resume text and vectorize
    raw_text = parse_pdf(resume_pdf_path)
    info = extract_resume_info('uploaded_resume', raw_text)
    text = ' '.join([info.get(k, '') for k in ['title','summary','skills','experience','education']])
    resume_vec = vectorize_text_glove(text, glove_embeddings)

    # 2. Compute cosine similarities via dot product
    dot = FILTERED_VECTORS.dot(resume_vec)  # O(n*d)
    r_norm = np.linalg.norm(resume_vec)
    similarity = dot / (FILTERED_NORMS * r_norm)

    # 3. Salary scoring
    r_salary = info.get('salary', 60000)
    sal_score = 1 - np.abs(FILTERED_SALARIES - r_salary) / (150000-30000)
    sal_score = np.clip(sal_score, 0, 1)

    # 4. Combine
    combined = lambda_pref * similarity + (1 - lambda_pref) * sal_score
    top_idx = np.argsort(combined)[::-1][:top_k]

    # 5. Select top job_ids and DataFrame rows
    top_ids = FILTERED_INDEX[top_idx]
    top_jobs = job_df.loc[top_ids]
    
    return top_jobs.copy(), similarity[top_idx], sal_score[top_idx], combined[top_idx]