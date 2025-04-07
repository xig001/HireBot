import nltk
nltk.download("punkt") 
import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import pickle
from model import recommend_jobs
from resume_parser import parse_pdf, extract_resume_info, vectorize_text_glove, load_glove_embeddings

st.set_page_config(page_title="HireBot – Job Recommendation System", layout="centered")

# --- UI Style ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body {
        font-family: 'Inter', sans-serif;
        background-image: url('https://images.unsplash.com/photo-1519389950473-47ba0277781c');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 3rem 2.5rem;
        border-radius: 16px;
        max-width: 720px;
        margin: 5rem auto;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        padding: 0.6rem 1.4rem;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">HireBot</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload your resume and receive personalized job recommendations instantly.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    # Direct download links from Google Drive
    glove_url = "https://drive.google.com/uc?export=download&id=1tytPPZiwriSzVL6br3sc3ggcnF9vXgei"
    job_url = "https://drive.google.com/uc?export=download&id=1zQdu6JIFwmXcU2yEjLL64TnHCDnqKn03"

    # Download and load job.pkl
    job_response = requests.get(job_url)
    with open("job.pkl", "wb") as f:
        f.write(job_response.content)
    job_df = pd.read_pickle("job.pkl")

    # Download and load GloVe
    glove_response = requests.get(glove_url)
    with open("glove.6B.100d.txt", "wb") as f:
        f.write(glove_response.content)
    glove = load_glove_embeddings("glove.6B.100d.txt")

    return job_df, glove

job_df, glove_embeddings = load_resources()

with st.form("upload_form"):
    uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf"])
    submit_button = st.form_submit_button("🔍 Get Recommendations")

if uploaded_file and submit_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        resume_path = temp_file.name

    with st.spinner("Analyzing resume..."):
        top_jobs, sim_scores, sal_scores, comb_scores = recommend_jobs(resume_path, job_df, glove_embeddings)

        st.subheader("🎯 Your Recommended Jobs")
        for i, row in top_jobs.iterrows():
            with st.expander(f"📌 {row['job_title']} at {row['company_name']}"):
                st.markdown(f"**Location**: {row.get('location', 'N/A')}")
                st.markdown(f"**Salary**: ${row['salary']:,}")
                st.markdown(f"**Similarity Score**: {sim_scores[i - top_jobs.index[0]]:.4f}")
                st.markdown(f"**Salary Match Score**: {sal_scores[i - top_jobs.index[0]]:.4f}")
                st.markdown(f"**Total Match Score**: {comb_scores[i - top_jobs.index[0]]:.4f}")
elif submit_button:
    st.warning("⚠️ Please upload a resume.")
