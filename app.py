import nltk
nltk.download('punkt_tab')
import os
import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import pickle
import requests
from model import recommend_jobs
from resume_parser import parse_pdf, extract_resume_info, vectorize_text_glove, load_glove_embeddings

raw_url = "https://www.dropbox.com/scl/fi/rjohtc1r5rrmy7m239mx1/filtered_job_details.csv?rlkey=0bloxkyefpsswlky1rugccoll&st=5e323yen&dl=1"
job_details_df = pd.read_csv(raw_url)

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
    glove_url = "https://drive.google.com/uc?export=download&id=1tytPPZiwriSzVL6br3sc3ggcnF9vXgei"
    job_url = "https://drive.google.com/uc?export=download&id=19Fd-HuXWu8Fq9W81HUp-ZW5yMk_eqthK"

    job_response = requests.get(job_url)
    with open("job.pkl", "wb") as f:
        f.write(job_response.content)
    job_df = pd.read_pickle("job.pkl")

    glove_response = requests.get(glove_url)
    with open("glove.6B.100d.txt", "wb") as f:
        f.write(glove_response.content)
    glove = load_glove_embeddings("glove.6B.100d.txt")

    return job_df, glove

job_df, glove_embeddings = load_resources()

# Add filters in sidebar
st.sidebar.header("Filter Jobs")
selected_location = st.sidebar.selectbox("Location", ["All"] + sorted(job_details_df["location"].dropna().unique()))
selected_work_type = st.sidebar.selectbox("Work Type", ["All"] + sorted(job_details_df["formatted_work_type"].dropna().unique()))

# Filter job_df
filtered_job_df = job_df.copy()
filtered_job_details_df = job_details_df.copy()

if selected_location != "All":
    filtered_job_details_df = filtered_job_details_df[filtered_job_details_df["location"] == selected_location]
if selected_work_type != "All":
    filtered_job_details_df = filtered_job_details_df[filtered_job_details_df["formatted_work_type"] == selected_work_type]

filtered_job_ids = filtered_job_details_df["job_id"].astype(str).tolist()
filtered_job_df = filtered_job_df[filtered_job_df["job_id"].astype(str).isin(filtered_job_ids)]

with st.form("upload_form"):
    uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf"])
    submit_button = st.form_submit_button("🔍 Get Recommendations")

if uploaded_file and submit_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        resume_path = temp_file.name

    with st.spinner("Analyzing resume..."):
        top_jobs, sim_scores, sal_scores, comb_scores = recommend_jobs(resume_path, filtered_job_df, glove_embeddings)

        top_jobs["job_id"] = top_jobs["job_id"].astype(str)
        filtered_job_details_df["job_id"] = filtered_job_details_df["job_id"].astype(str)
        matched_job_details = filtered_job_details_df[filtered_job_details_df["job_id"].isin(top_jobs["job_id"])].reset_index(drop=True)

        st.subheader("🎯 Your Recommended Jobs")

        for i, row in matched_job_details.iterrows():
            with st.expander(f"📌 {row['title']} at {row['company_name']}"):
                st.markdown(f"**📍 Location**: {row.get('location', 'N/A')}")
                st.markdown(f"**🕒 Work Type**: {row.get('formatted_work_type', 'N/A')}")
                salary = row.get('max_salary')
                if pd.notna(salary):
                    st.markdown(f"**💰 Salary**: ${int(salary):,}")
                else:
                    st.markdown("**💰 Salary**: Not specified")
                st.markdown(f"**🔗 Apply Here**: [Application Link]({row.get('application_url', '#')})")
                st.markdown("**📜 Job Description**:")
                st.text_area("", row.get('description', 'No description available.'), height=200)

elif submit_button:
    st.warning("⚠️ Please upload a resume.")
