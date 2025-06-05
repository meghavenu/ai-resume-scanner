import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pickle
import requests
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_lottie import st_lottie
import numpy as np

st.set_page_config(page_title="AI Resume Scanner", layout="wide", page_icon="📄")

# Load ML models and NLP
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

# Load Lottie animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

scan_lottie = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_u8o7BL.json")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
    st.title("Resume Scanner Pro")
    st.markdown("🔍 Upload resumes & get AI-powered feedback.")
    st.markdown("---")
    st.info("Built with 💼 ML, 🧠 NLP, and 🎨 Lottie")
    st.markdown("[Contact: LinkedIn](https://linkedin.com)")

# Dark theme styles
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #57c1eb;
    }
    .st-bb {
        background-color: #0e1117;
    }
    textarea, .stTextArea>div>div>textarea {
        background-color: #1e242c;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with animation & title
col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(scan_lottie, height=200)
with col2:
    st.title("AI Resume Scanner")
    st.write("Get instant feedback, job category prediction, resume score, and improvement tips.")

# Upload resumes and job description input
resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)")

# Text extraction functions
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

# Extract sections by header keywords
def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {}
    for h in headers:
        regex = re.compile(rf"{h}[:\n](.*?)(?=\n[A-Z][a-z]+:|\Z)", re.DOTALL | re.IGNORECASE)
        match = regex.search(text)
        if match:
            sections[h] = match.group(1).strip()
        else:
            sections[h] = "Not found"
    return sections

# Generate word cloud plot
def generate_wordcloud(text):
    wc = WordCloud(background_color="#0e1117", colormap="Blues", width=400, height=200).generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Pie chart for section word counts
def plot_section_pie(sections):
    counts = {k: len(v.split()) if v != "Not found" else 0 for k, v in sections.items()}
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

# Compute resume score (dummy function - adjust with your logic)
def compute_resume_score(text):
    score = min(100, len(text) // 10)
    return score

# AI job role prediction
def predict_job_role(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return pred[0]

# Similarity with job description
def compute_match_score(resume_text, job_desc_text):
    if not job_desc_text.strip():
        return None
    docs = [resume_text, job_desc_text]
    tfidf = TfidfVectorizer().fit_transform(docs)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

# Strengths and improvements (dummy example)
def analyze_strengths_improvements(text):
    strengths = []
    improvements = []
    if "lead" in text.lower():
        strengths.append("Leadership mentioned")
    else:
        improvements.append("Add leadership experience if applicable")
    if "python" in text.lower():
        strengths.append("Python skill detected")
    else:
        improvements.append("Mention relevant technical skills")
    return strengths, improvements

# Process resumes
if resumes:
    for file in resumes:
        st.markdown(f"### Analyzing: {file.name}")
        text = extract_text(file)
        cleaned = clean_text(text)
        sections = extract_sections(text)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Section-wise Analysis")
            for sec, content in sections.items():
                st.markdown(f"**{sec}:**")
                st.write(content[:300] + ("..." if len(content) > 300 else ""))
        with col2:
            st.subheader("Visual Insights")
            generate_wordcloud(cleaned)
            plot_section_pie(sections)

        st.subheader("AI Insights")
        job_role = predict_job_role(cleaned)
        st.markdown(f"**Predicted Job Role:** {job_role}")

        score = compute_resume_score(cleaned)
        st.markdown(f"**Resume Score:** {score} / 100")

        if job_desc.strip():
            match_score = compute_match_score(cleaned, job_desc)
            st.markdown(f"**Match Score with Job Description:** {match_score}%")

        strengths, improvements = analyze_strengths_improvements(cleaned)
        st.markdown("**Strengths:**")
        for s in strengths:
            st.success(s)
        st.markdown("**Areas for Improvement:**")
        for imp in improvements:
            st.warning(imp)

        st.markdown("---")
else:
    st.info("Upload one or more resumes to start the analysis.")
