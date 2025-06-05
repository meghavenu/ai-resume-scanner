AI Resume Scanner - Enhanced Version

import streamlit as st import PyPDF2 import docx2txt import matplotlib.pyplot as plt from wordcloud import WordCloud import re import spacy import pickle import numpy as np from sklearn.metrics.pairwise import cosine_similarity from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("resume_model.pkl", "rb")) vectorizer = pickle.load(open("vectorizer.pkl", "rb")) nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Scanner Pro", layout="wide") st.title("📄 AI Resume Scanner Pro")

resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True) job_desc = st.text_area("Paste Job Description (Optional)")

def extract_text(file): text = "" if file.type == "application/pdf": reader = PyPDF2.PdfReader(file) for page in reader.pages: text += page.extract_text() elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": text = docx2txt.process(file) return text

def clean_resume(text): return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

def extract_sections(text): sections = {} headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"] for header in headers: pattern = re.compile(header + r'[:\n][\s\S]{0,500}', re.IGNORECASE) match = pattern.search(text) if match: sections[header] = match.group() return sections

def resume_match_score(resume, jd): vect = TfidfVectorizer() tfidf = vect.fit_transform([resume, jd]) return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def generate_wordcloud(text): wc = WordCloud(width=800, height=300, background_color='black', colormap='Blues').generate(text) plt.imshow(wc, interpolation='bilinear') plt.axis("off") st.pyplot(plt)

def smart_suggestions(category): suggestions = { "Data Scientist": "NumPy, Pandas, Deep Learning, Statistics", "Software Engineer": "Git, REST APIs, Docker, System Design", "DevOps": "Jenkins, CI/CD, Docker, Monitoring", "Product Manager": "Agile, Roadmapping, User Stories, KPIs", "AI Engineer": "TensorFlow, PyTorch, NLP, Computer Vision" } return suggestions.get(category, "Consider enhancing your skills based on your job interest.")

def detect_strengths(text): strengths = [] if "led" in text or "managed" in text: strengths.append("Leadership") if "python" in text or "sql" in text: strengths.append("Technical Skills") if "team" in text or "collaborated" in text: strengths.append("Teamwork") return strengths

def improvement_suggestions(text): suggestions = [] if len(text.split()) < 150: suggestions.append("Resume is too short. Add more content.") if "project" not in text.lower(): suggestions.append("Add a Projects section to highlight practical work.") if len(set(re.findall(r'\b(\w+)\b', text.lower()))) < 100: suggestions.append("Consider adding more vocabulary diversity.") return suggestions

def section_completion_score(sections): all_sections = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"] return int((len(sections) / len(all_sections)) * 100)

if resumes: for file in resumes: text = extract_text(file) cleaned = clean_resume(text) st.subheader(f"📄 {file.name}")

sections = extract_sections(cleaned)
    st.write("### 📚 Extracted Sections:")
    for sec, content in sections.items():
        st.write(f"**{sec}**: {content[:300]}...")

    st.write("### 🌐 WordCloud")
    generate_wordcloud(cleaned)

    if job_desc:
        score = resume_match_score(cleaned, job_desc)
        st.write(f"### 🔗 Resume-Job Match Score: **{score}%**")

    vector_input = vectorizer.transform([cleaned])
    predicted_category = model.predict(vector_input)[0]
    st.write(f"### 🔍 Predicted Job Category: **{predicted_category}**")

    strengths = detect_strengths(cleaned)
    st.write(f"### 💪 Detected Strengths: {', '.join(strengths) if strengths else 'No strong traits found.'}")

    improve = improvement_suggestions(cleaned)
    st.write("### 🔧 Suggestions to Improve Resume:")
    for s in improve:
        st.write("- " + s)

    st.write(f"### 🧭 Smart Skill Suggestions: {smart_suggestions(predicted_category)}")

    section_score = section_completion_score(sections)
    st.write(f"### 🧩 Section Completion Score: **{section_score}%**")

    overall_score = (score if job_desc else 70) * 0.4 + section_score * 0.3 + len(strengths) * 10
    overall_score = min(round(overall_score), 100)
    st.write(f"### 🏅 Overall Resume Score: **{overall_score}/100**")

else: st.info("Upload a resume to begin analysis.")

