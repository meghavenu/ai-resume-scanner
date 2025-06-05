import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pickle
import requests
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Resume Scanner Pro", layout="wide")

# Load models
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

scan_lottie = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_u8o7BL.json")

# Inject Google Fonts and professional CSS
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f9fafb;
            color: #1f2937;
            margin: 0;
            padding: 0;
        }
        header {
            padding: 1rem 2rem;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            font-size: 1.8rem;
            text-align: center;
            letter-spacing: 1.2px;
        }
        .container {
            max-width: 1100px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .resume-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgb(0 0 0 / 0.1);
            padding: 1.5rem;
            margin-bottom: 2rem;
            transition: box-shadow 0.3s ease;
        }
        .resume-card:hover {
            box-shadow: 0 12px 24px rgb(0 0 0 / 0.15);
        }
        .score-bar {
            height: 14px;
            border-radius: 7px;
            background-color: #e0e7ff;
            margin-top: 4px;
        }
        .score-fill {
            height: 100%;
            background-color: #2563eb;
            border-radius: 7px;
            transition: width 0.5s ease;
        }
        .subheader {
            font-weight: 600;
            font-size: 1.2rem;
            margin-top: 1rem;
        }
        .wordcloud-container {
            max-width: 450px;
            margin-top: 1rem;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1.5rem;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #d1d5db;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        .comparison-table th {
            background-color: #2563eb;
            color: white;
        }
        .highlight-best {
            background-color: #bbf7d0 !important;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<header>Resume Scanner Pro</header>", unsafe_allow_html=True)

st.markdown("<div class='container'>", unsafe_allow_html=True)

# Upload section with Lottie
col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(scan_lottie, height=180)
with col2:
    st.title("Upload Your Resume(s)")
    st.write("Get AI-powered insights, match scores, and compare resumes easily.")

resumes = st.file_uploader("Select Resume files (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)", height=120)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {}
    for h in headers:
        match = re.search(h + r'[:\n][\s\S]{0,500}', text, re.IGNORECASE)
        if match:
            sections[h] = match.group()
    return sections

def match_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

def generate_wordcloud(text):
    wc = WordCloud(width=400, height=200, background_color='white', colormap='Blues').generate(text)
    fig, ax = plt.subplots(figsize=(4,2))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

if resumes:
    for resume in resumes:
        text = extract_text(resume)
        cleaned = clean_text(text)
        sections = extract_sections(cleaned)
        predicted_role = model.predict(vectorizer.transform([cleaned]))[0]
        score = match_score(cleaned, job_desc)

        st.markdown(f"<div class='resume-card'>", unsafe_allow_html=True)

        st.markdown(f"### 📄 {resume.name}")
        st.markdown(f"**Predicted Job Role:** {predicted_role}")
        st.markdown(f"**Resume Length:** {len(cleaned.split())} words")

        st.markdown("**JD Match Score:**")
        pct = score
        st.markdown(f"""
            <div class="score-bar">
                <div class="score-fill" style="width:{pct}%;"></div>
            </div>
            <span>{pct}% match</span>
        """, unsafe_allow_html=True)

        st.markdown("**Detected Sections:**")
        for sec in sections:
            st.markdown(f"- {sec}")

        st.markdown("**Word Cloud:**")
        generate_wordcloud(cleaned)

        st.markdown("</div>", unsafe_allow_html=True)

# Resume Comparison Table
if resumes and len(resumes) > 1:
    st.markdown("---")
    st.subheader("📊 Resume Comparison")

    data = []
    for resume in resumes:
        text = extract_text(resume)
        cleaned = clean_text(text)
        sections = extract_sections(cleaned)
        predicted_role = model.predict(vectorizer.transform([cleaned]))[0]
        score = match_score(cleaned, job_desc)
        total_score = score + len(sections)*10 + len(cleaned.split())/10
        data.append({
            "Name": resume.name,
            "Role": predicted_role,
            "Length (words)": len(cleaned.split()),
            "JD Match (%)": score,
            "Sections": len(sections),
            "Total Score": round(total_score, 2)
        })

    df = pd.DataFrame(data)
    best_idx = df['Total Score'].idxmax()

    def highlight_best(row):
        return ['background-color: #bbf7d0; font-weight: 700' if i == best_idx else '' for i in row.index]

    st.dataframe(df.style.apply(highlight_best, axis=1))

    st.success(f"🏆 Best Resume: **{df.loc[best_idx, 'Name']}** based on combined score.")

st.markdown("</div>", unsafe_allow_html=True)
