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
raw_url = "www.dropbox.com/scl/fi/rjohtc1r5rrmy7m239mx1/filtered_job_details.csv?rlkey=0bloxkyefpsswlky1rugccoll&st=5e323yen&raw=1"
# Load the CSV file
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
    # Direct download links from Google Drive
    glove_url = "https://drive.google.com/uc?export=download&id=1tytPPZiwriSzVL6br3sc3ggcnF9vXgei"
    job_url = "https://drive.google.com/uc?export=download&id=19Fd-HuXWu8Fq9W81HUp-ZW5yMk_eqthK"

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
st.write("📄 Job Details Sample:")
st.dataframe(job_details_df.head())
with st.form("upload_form"):
    uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf"])
    submit_button = st.form_submit_button("🔍 Get Recommendations")

if uploaded_file and submit_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        resume_path = temp_file.name

    with st.spinner("Analyzing resume..."):
        top_jobs, sim_scores, sal_scores, comb_scores = recommend_jobs(resume_path, job_df, glove_embeddings)
        
        top_jobs["job_id"] = top_jobs["job_id"].astype(str)
        job_details_df[job_details_df.columns[0]]
        top_job_ids = top_jobs["job_id"].tolist()
        matched_job_details = job_details_df[job_details_df[job_details_df.columns[0]].isin(top_job_ids)].reset_index(drop=True)
        
        st.subheader("🎯 Your Recommended Jobs")
        
        st.dataframe(matched_job_details)
        #for i, row in matched_job_details.iterrows():
         #   with st.expander(f"📌 {row[2]} at {row[1]}"):  # title at index 2, company_name at index 1
          #      st.markdown(f"**📍 Location**: {row[6]}")  # location at index 6
           #     st.markdown(f"**🕒 Work Type**: {row[11]}")  # formatted_work_type at index 11
            #    st.markdown(f"**💰 Salary**: ${int(row[4]):,}")  # max_salary at index 4
              #  st.markdown(f"**🔗 Apply Here**: [Application Link]({row[16]})")  # application_url at index 16
             #   st.markdown("**📝 Job Description**:")
              #  st.text_area("", row[3], height=200)  # description at index

elif submit_button:
    st.warning("⚠️ Please upload a resume.")
