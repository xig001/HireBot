import nltk
nltk.download('punkt_tab')
import os
import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import requests
from model import recommend_jobs_2
from resume_parser import load_glove_embeddings

st.set_page_config(page_title="HireBot – Job Recommendation System", layout="centered")

raw_url = "https://www.dropbox.com/scl/fi/rjohtc1r5rrmy7m239mx1/filtered_job_details.csv?rlkey=0bloxkyefpsswlky1rugccoll&st=5e323yen&dl=1"

@st.cache_data
def load_job_details(url):
    df = pd.read_csv(url)
    # Ensure consistent types
    df['job_id'] = df['job_id'].astype(str)
    # Return filtered details
    return df

job_details_df = load_job_details(raw_url)  # Cached: only reads CSV once per session




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

    # Download job.pkl containing precomputed glove vectors and salaries
    job_response = requests.get(job_url)
    with open("job.pkl", "wb") as f:
        f.write(job_response.content)
    job_df = pd.read_pickle("job.pkl")
    job_df['job_id'] = job_df['job_id'].astype(str)
    job_df = job_df.set_index('job_id')

    # Extract precomputed vectors and compute norms once
    glove_cols = [c for c in job_df.columns if c.startswith('glove_')]
    job_vectors = job_df[glove_cols].to_numpy()  # precomputed glove vectors
    job_norms = np.linalg.norm(job_vectors, axis=1)  # norms for cosine denom

    # Salaries array for scoring
    if 'salary' in job_df.columns:
        salaries = job_df['salary']
    else:
        max_sal = job_df['max_salary']
        # Identify missing entries and hourly salaries
        missing_mask = max_sal.isna()
        sal = max_sal.copy().fillna(0)
        # Hourly wages assumed if < 1000: convert to annual (40h/week * 52 weeks)
        hourly_mask = sal < 1000
        sal[hourly_mask] = sal[hourly_mask] * 40 * 52
        # For original NA entries, generate random annual salary between 30k and 150k
        if missing_mask.any():
            sal[missing_mask] = np.random.randint(30000, 150001, size=missing_mask.sum())
        salaries = sal
    job_salaries = salaries.to_numpy()

    # Return all needed resources
    glove = load_glove_embeddings("glove.6B.100d.txt")
    return job_df, job_vectors, job_norms, job_salaries, glove

job_df, JOB_VECTORS, JOB_NORMS, JOB_SALARIES, glove_embeddings = load_resources()

# Add filters in sidebar
st.sidebar.header("Filter Jobs")
selected_location = st.sidebar.selectbox("Location", ["All"] + sorted(job_details_df["location"].dropna().unique()))
selected_work_type = st.sidebar.selectbox("Work Type", ["All"] + sorted(job_details_df["formatted_work_type"].dropna().unique()))

# Apply boolean masks
def apply_filters(details_df):
    mask_loc = (details_df['location'] == selected_location) if selected_location != 'All' else pd.Series(True, index=details_df.index)
    mask_work = (details_df['formatted_work_type'] == selected_work_type) if selected_work_type != 'All' else pd.Series(True, index=details_df.index)
    return details_df[mask_loc & mask_work]



filtered_details = apply_filters(job_details_df)
# Filter job_df by index
filtered_ids = filtered_details['job_id'].astype(str)
filtered_mask = job_df.index.isin(filtered_ids)

# Pre-slice numpy arrays according to filter
FILTERED_VECTORS = JOB_VECTORS[filtered_mask]
FILTERED_NORMS = JOB_NORMS[filtered_mask]
FILTERED_SALARIES = JOB_SALARIES[filtered_mask]
FILTERED_INDEX = job_df.index[filtered_mask]

with st.form("upload_form"):
    uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf"])
    submit_button = st.form_submit_button("🔍 Get Recommendations")

if uploaded_file and submit_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        resume_path = temp_file.name

    with st.spinner("Analyzing resume..."):
        top, sim, sal, comb = recommend_jobs_2(resume_path, job_df, glove_embeddings, FILTERED_VECTORS, FILTERED_NORMS, FILTERED_SALARIES, FILTERED_INDEX, 0.75, 3)

        st.subheader("🎯 Your Recommended Jobs")

        for job_id in top.index:
            row = filtered_details.set_index('job_id').loc[job_id]
            with st.expander(f"📌 {row['title']} at {row['company_name']}"):
                st.markdown(f"**📍 Location**: {row.get('location','N/A')}")
                st.markdown(f"**🕒 Work Type**: {row.get('formatted_work_type','N/A')}")
                salv = row.get('max_salary')
                st.markdown(f"**💰 Salary**: ${int(salv):,}" if pd.notna(salv) else "**💰 Salary**: Not specified")
                st.markdown(f"**🔗 Apply Here**: [Link]({row.get('application_url','#')})")
                st.text_area('', row.get('description','No description.'), height=200)
elif submit_button:
    st.warning("⚠️ Please upload a resume.")
