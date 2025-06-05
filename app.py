import streamlit as st
import PyPDF2
import docx2txt
import re
import pickle
import requests
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie

# Setup page
st.set_page_config(page_title="AI Resume Scanner Pro", layout="wide")

# Load models & NLP
model = pickle.load(open("resume_model.pkl", "rb"))        # Your trained job role classifier
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))    # Your TF-IDF vectorizer
nlp = spacy.load("en_core_web_sm")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_u8o7BL.json")

# Styling CSS
st.markdown("""
<style>
body {
    background-color: #f9fbfd;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333;
}
h1, h2, h3 {
    color: #1f2937;
}
.resume-section {
    background: white;
    padding: 15px;
    margin-bottom: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.st-bb {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
    st.title("Resume Scanner Pro")
    st.markdown("Upload your resumes and job description to get tailored AI-powered insights.")
    st.markdown("---")
    st.info("Powered by Machine Learning, NLP, and Beautiful Visuals")
    st.markdown("Contact: [LinkedIn](https://linkedin.com)")

st_lottie(lottie_animation, height=180, key="resume_animation")

st.title("📋 AI Resume Scanner")

uploaded_files = st.file_uploader("Upload Resume(s) (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description (optional)", height=150)

# Helper Functions

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9.,;:\-() ]+', ' ', text)
    return text.strip()

def extract_sections(text):
    sections = {}
    patterns = {
        "Education": r"(Education|Academic Background|Qualifications)(.*?)(Experience|Skills|Projects|Certifications|Achievements|$)",
        "Experience": r"(Experience|Work History|Professional Experience)(.*?)(Education|Skills|Projects|Certifications|Achievements|$)",
        "Skills": r"(Skills|Technical Skills|Core Competencies)(.*?)(Education|Experience|Projects|Certifications|Achievements|$)",
        "Projects": r"(Projects|Relevant Projects)(.*?)(Education|Experience|Skills|Certifications|Achievements|$)",
        "Certifications": r"(Certifications|Licenses)(.*?)(Education|Experience|Skills|Projects|Achievements|$)",
        "Achievements": r"(Achievements|Awards)(.*?)(Education|Experience|Skills|Projects|Certifications|$)"
    }
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(2).strip()
            if len(content) > 20:
                sections[section] = content
    return sections

def extract_skills(text):
    # Simplified skills extraction using a keyword list - ideally use a better method or external API
    skills_keywords = [
        'python', 'java', 'c++', 'sql', 'machine learning', 'deep learning', 'nlp', 'data analysis', 'excel',
        'power bi', 'tableau', 'communication', 'teamwork', 'project management', 'leadership'
    ]
    text_lower = text.lower()
    found_skills = [skill for skill in skills_keywords if skill in text_lower]
    return found_skills

def match_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0
    tfidf = TfidfVectorizer(stop_words='english')
    docs = [resume_text, jd_text]
    matrix = tfidf.fit_transform(docs)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

def generate_wordcloud(text):
    wc = WordCloud(width=700, height=350, background_color='white', colormap='Blues').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def generate_suggestions(sections, skills, jd_text):
    suggestions = []

    if "Education" not in sections:
        suggestions.append("Add a clear Education section with your academic qualifications.")
    else:
        suggestions.append("Your Education section is present and well-defined.")

    if "Experience" not in sections:
        suggestions.append("Consider adding detailed professional Experience to highlight your work history.")
    else:
        suggestions.append("Your Experience section is informative.")

    if "Skills" not in sections or len(skills) < 3:
        suggestions.append("Expand your Skills section with relevant technical and soft skills.")
    else:
        suggestions.append(f"Skills section is good and includes: {', '.join(skills)}.")

    if jd_text.strip():
        jd_skills = extract_skills(jd_text)
        missing_skills = set(jd_skills) - set(skills)
        if missing_skills:
            suggestions.append(f"Consider adding these key skills matching the job description: {', '.join(missing_skills)}.")
        else:
            suggestions.append("Your skills match well with the job description.")

    # Style & readability
    suggestions.append("Maintain concise bullet points and use action verbs to describe your achievements.")
    suggestions.append("Proofread to avoid grammatical errors and ensure consistent formatting.")

    return suggestions

# Main Logic
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📄 Analyzing: {uploaded_file.name}")

        raw_text = extract_text(uploaded_file)
        cleaned_text = clean_text(raw_text)
        sections = extract_sections(cleaned_text)
        skills = extract_skills(cleaned_text)

        predicted_role = model.predict(vectorizer.transform([cleaned_text]))[0]
        jd_score = match_score(cleaned_text, job_description)

        # Display sections
        with st.container():
            st.markdown("### 🧾 Extracted Sections")
            if sections:
                for sec, content in sections.items():
                    st.markdown(f"**{sec}:**")
                    st.write(content[:500] + ("..." if len(content) > 500 else ""))
            else:
                st.info("No clear sections detected. Consider structuring your resume with headers like Education, Experience, Skills.")

        # Display skills
        st.markdown("### 💼 Skills Detected")
        if skills:
            st.write(", ".join(skills))
        else:
            st.info("No skills detected. Make sure to include a dedicated Skills section.")

        # Predicted Role & Match Score
        st.markdown("### 🎯 Job Role Prediction & JD Match")
        st.success(f"Predicted Job Role: {predicted_role}")
        if job_description.strip():
            st.info(f"Job Description Match Score: {jd_score}%")
        else:
            st.info("No job description provided, skipping match score.")

        # Suggestions
        st.markdown("### 🛠️ Suggestions to Improve Your Resume")
        suggestions = generate_suggestions(sections, skills, job_description)
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        # Word Cloud
        st.markdown("### ☁️ Word Cloud Visualization")
        generate_wordcloud(cleaned_text)

        # Resume length
        st.markdown(f"### 📏 Resume Length: {len(cleaned_text.split())} words")

    # Resume comparison if multiple
    if len(uploaded_files) > 1:
        st.markdown("---")
        st.subheader("📊 Resume Comparison")

        comparison_rows = []
        for file in uploaded_files:
            txt = clean_text(extract_text(file))
            sec = extract_sections(txt)
            skl = extract_skills(txt)
            role = model.predict(vectorizer.transform([txt]))[0]
            score = match_score(txt, job_description
