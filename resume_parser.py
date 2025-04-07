
import re
import numpy as np
from collections import defaultdict
import PyPDF2
import nltk
import os
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

# 👇 Again, tell NLTK where to look for punkt
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

SECTION_ALIASES = {
    "summary": ["summary", "highlights", "profile", "overview"],
    "skills": ["skills", "technical skills", "core competencies"],
    "experience": ["experience", "professional experience", "work history", "employment history"],
    "education": ["education", "academic background", "qualifications"]
}

STOPWORDS = {'the', 'is', 'am', 'are', 'was', 'were', 'this', 'that', 'of', 'at', 'to', 'in', 'and', 'for', 'a', 'with', 'as', 'all', 'on', 'state', 'city', 'name', 'company', 'i', 'by', 'within', 'skill', 'skills', 'where', 'use', 'using', 'include'}

def preprocess_text_light(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

def detect_sections(raw_text):
    sections = defaultdict(str)
    raw_text = raw_text.strip()
    pattern = '|'.join([rf"\b{item}\b" for sublist in SECTION_ALIASES.values() for item in sublist])
    section_pattern = re.compile(rf"(?P<header>{pattern})", re.IGNORECASE)
    matches = list(section_pattern.finditer(raw_text))
    if not matches:
        return sections
    for i, match in enumerate(matches):
        section_name = match.group("header").strip().lower()
        section_key = next((k for k, v in SECTION_ALIASES.items() if section_name in v), None)
        if section_key:
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw_text)
            sections[section_key] = raw_text[start:end].strip()
    for key in SECTION_ALIASES.keys():
        sections.setdefault(key, "")
    return sections

def extract_resume_info(resume_id, raw_text):
    sections = detect_sections(raw_text)
    processed = {k: preprocess_text_light(v) for k, v in sections.items()}
    title_line = raw_text.split("\n")[0].strip().split("  ")[0]
    title = preprocess_text_light(title_line)
    salary = np.random.randint(30000, 150001)
    return {
        "resume_id": resume_id,
        "title": title,
        "summary": processed.get("summary", ""),
        "skills": processed.get("skills", ""),
        "experience": processed.get("experience", ""),
        "education": processed.get("education", ""),
        "salary": salary
    }

def parse_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text = f"Error parsing PDF: {e}"
    return text

def vectorize_text_glove(text, glove_embeddings, dim=100):
    words = text.split()
    vecs = [glove_embeddings[w] for w in words if w in glove_embeddings]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

def load_glove_embeddings(filepath, dim=100):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue  # skip corrupted or incomplete lines
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype="float32")
                embeddings[word] = vec
            except ValueError:
                continue  # skip if numbers are malformed
    return embeddings
