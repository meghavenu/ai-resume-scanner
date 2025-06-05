import streamlit as st
import PyPDF2
import docx2txt
import re
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.set_page_config(page_title="Resume Scanner Pro", layout="wide")

# Load NLP model and ML models
nlp = spacy.load("en_core_web_sm")
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return ""

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {}
    for header in headers:
        pattern = rf"{header}.*?(?=(Education|Experience|Projects|Skills|Certifications|Achievements|$))"
        match = re.search(pattern, text, re.I | re.S)
        if match:
            sections[header] = match.group().strip()
    return sections

def extract_skills(text):
    # Simple skills keyword list
    skills_keywords = ['python', 'java', 'machine learning', 'data analysis', 'sql', 'communication', 'project management']
    found_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return found_skills

def job_role_prediction(text):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    return pred

def jd_match_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform([resume_text, jd_text])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

def generate_wordcloud(text):
    wc = WordCloud(width=400, height=200, background_color='white').generate(text)
    plt.figure(figsize=(6,3))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def generate_suggestions(sections, skills, jd_skills):
    suggestions = []
    # Check missing sections
    required_sections = ["Education", "Experience", "Skills"]
    for sec in required_sections:
        if sec not in sections:
            suggestions.append(f"Consider adding a **{sec}** section for completeness.")

    # Skill gaps
    missing_skills = set(jd_skills) - set(skills)
    if missing_skills:
        suggestions.append(f"Add or highlight these skills to improve job match: {', '.join(missing_skills)}.")

    if not suggestions:
        suggestions.append("Your resume looks well structured and aligned!")

    return suggestions

# UI

st.title("Resume Scanner Pro — AI-powered Resume Analysis")
st.write("Upload your resume(s) and paste the job description below to get detailed insights and suggestions.")

uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", accept_multiple_files=True)
job_description = st.text_area("Paste Job Description here", height=150)

if uploaded_files and st.button("Analyze Resumes"):

    jd_skills = extract_skills(job_description)

    results = []
    for file in uploaded_files:
        raw_text = extract_text(file)
        clean_resume_text = clean_text(raw_text)
        sections = extract_sections(raw_resume_text := raw_text.lower())
        skills = extract_skills(clean_resume_text)
        role = job_role_prediction(clean_resume_text)
        match_score = jd_match_score(clean_resume_text, job_description)

        suggestions = generate_suggestions(sections, skills, jd_skills)

        results.append({
            "name": file.name,
            "role": role,
            "match_score": match_score,
            "sections": sections,
            "skills": skills,
            "suggestions": suggestions,
            "text": clean_resume_text
        })

    for res in results:
        st.header(f"Resume: {res['name']}")
        st.subheader(f"Predicted Job Role: {res['role']}")
        st.markdown(f"**Job Description Match Score:** {res['match_score']}%")
        st.markdown("### Detected Sections:")
        for sec in res["sections"]:
            st.markdown(f"- **{sec}**")
        st.markdown("### Extracted Skills:")
        st.markdown(", ".join(res["skills"]) if res["skills"] else "No skills detected")
        st.markdown("### Suggestions to Improve Your Resume:")
        for s in res["suggestions"]:
            st.markdown(f"- {s}")
        st.markdown("### Word Cloud:")
        generate_wordcloud(res["text"])
        st.markdown("---")

    if len(results) > 1:
        st.subheader("Resume Comparison")
        comp_df = pd.DataFrame({
            "Resume": [r["name"] for r in results],
            "Predicted Role": [r["role"] for r in results],
            "Match Score (%)": [r["match_score"] for r in results],
            "Skills Found": [", ".join(r["skills"]) for r in results],
            "Sections Count": [len(r["sections"]) for r in results]
        })
        st.dataframe(comp_df)
