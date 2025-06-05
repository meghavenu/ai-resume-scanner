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

# Set page configuration
st.set_page_config(page_title="AI Resume Scanner", layout="wide")

# Load models and vectorizers
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

# Load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

scan_lottie = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_u8o7BL.json")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f4f4;
        color: #333;
    }
    h1, h2, h3, h4 {
        color: #007BFF; /* A professional blue */
    }
    .st-bb {
        background-color: #ffffff; /* White background for better readability */
    }
    .main {
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
    st.title("Resume Scanner Pro")
    st.markdown("🔍 Upload resumes & get AI-powered feedback.")
    st.markdown("---")
    st.info("Built with 💼 ML, 🧠 NLP, and 🎨 Lottie")
    st.markdown("Contact: [LinkedIn](https://linkedin.com)")

# Main content
st.markdown("<h1 style='text-align: center;'>AI Resume Scanner</h1>", unsafe_allow_html=True)
st.write("Get instant feedback, job category prediction, resume score, and improvement tips.")

# File uploader and job description input
resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)")

# Function to extract text from files
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

# Function to clean text
def clean_text(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

# Function to extract sections from resume
def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {}
    for h in headers:
        match = re.search(h + r'[:\n][\s\S]{0,500}', text, re.IGNORECASE)
        if match:
            sections[h] = match.group()
    return sections

# Function to calculate match score
def match_score(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

# Function to generate word cloud
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Process uploaded resumes
if resumes:
    for resume in resumes:
        st.markdown("----")
        st.header(f"📄 {resume.name}")
        text = extract_text(resume)
        cleaned = clean_text(text)
        sections = extract_sections(cleaned)
        predicted_role = model.predict(vectorizer.transform([cleaned]))[0]
        score = match_score(cleaned, job_desc) if job_desc else 0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📌 Predicted Job Role")
            st.success(predicted_role)

            st.subheader("📊 JD Match Score")
            st.info(f"{score}% match")

            st.subheader("🧠 Resume Sections")
            for k in sections:
                st.markdown(f"- **{k}**")

        with col2:
            st.subheader("☁️ Word Cloud")
            generate_wordcloud(cleaned)

            st.subheader("🎯 Resume Length")
            st.write(f"{len(cleaned.split())} words")

# Resume Comparison Section
if len(resumes) >= 2:
    st.markdown("----")
    st.subheader("📊 Resume Comparison")
    comparison_data = []

    for resume in resumes:
        text = extract_text(resume)
        cleaned = clean_text(text)
        sections = extract_sections(cleaned)
        predicted_role = model.predict(vectorizer.transform([cleaned]))[0]
        score = match_score(cleaned, job_desc) if job_desc else 0
        total_score = score + len(sections) * 10 + len(cleaned.split()) / 10

        comparison_data.append({
            "Name": resume.name,
            "Predicted Role": predicted_role,
            "Length (words)": len(cleaned.split()),
            "JD Match (%)": score,
            "Sections": len(sections),
            "Total Score": round(total_score, 2)
        })

    df = pd.DataFrame(comparison_data).sort_values("Total Score", ascending=False).reset_index(drop=True)
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
    best_resume = df.iloc[0]["Name"]
    st.success(f"🏆 **Best Resume:** `{best_resume}` based on combined score of content, JD match, and structure.")
